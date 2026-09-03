#include "agent_tools.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

std::string quoted(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string error_json(const std::string& message, const std::string& path = "") {
    std::string result = "{";
    if (!path.empty()) {
        result += "\"path\":" + quoted(path) + ",";
    }
    result += "\"error\":" + quoted(message) + "}";
    return result;
}

int copy_result(const std::string& value, char* output, size_t output_size,
                size_t* required_size) {
    if (required_size == nullptr) return -1;
    *required_size = value.size() + 1;
    if (output == nullptr || output_size < *required_size) return 1;
    std::memcpy(output, value.c_str(), *required_size);
    return 0;
}

template <typename Function>
int run_json(Function function, char* output, size_t output_size,
             size_t* required_size) {
    // ctypes first asks for the required size and then supplies a buffer. Keep
    // the first result so state-changing tools are executed exactly once.
    thread_local std::string pending_result;
    try {
        if (output == nullptr || pending_result.empty()) {
            pending_result = function();
        }
        const int status = copy_result(pending_result, output, output_size, required_size);
        if (status == 0) pending_result.clear();
        return status;
    } catch (const std::exception& exc) {
        pending_result = error_json(exc.what());
        const int status = copy_result(pending_result, output, output_size, required_size);
        if (status == 0) pending_result.clear();
        return status;
    } catch (...) {
        pending_result = error_json("Unknown native tool error");
        const int status = copy_result(pending_result, output, output_size, required_size);
        if (status == 0) pending_result.clear();
        return status;
    }
}

fs::path from_utf8(const char* value) {
    if (value == nullptr) throw std::invalid_argument("A required string argument is null");
#ifdef _WIN32
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value, -1,
                                           nullptr, 0);
    if (length == 0) throw std::invalid_argument("Path is not valid UTF-8");
    std::wstring wide(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value, -1,
                        wide.data(), length);
    wide.pop_back();
    return fs::path(wide);
#else
    return fs::u8path(value);
#endif
}

std::string to_utf8(const fs::path& value) {
#ifdef _WIN32
    const std::wstring wide = value.wstring();
    if (wide.empty()) return "";
    const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                           wide.data(), static_cast<int>(wide.size()),
                                           nullptr, 0, nullptr, nullptr);
    if (length == 0) throw std::runtime_error("Unable to encode a Windows path as UTF-8");
    std::string result(static_cast<size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                        wide.data(), static_cast<int>(wide.size()),
                        result.data(), length, nullptr, nullptr);
    return result;
#elif defined(__cpp_lib_char8_t)
    const auto text = value.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
#else
    return value.u8string();
#endif
}

fs::path absolute_input_path(const char* value) {
    fs::path path = from_utf8(value);
    if (!path.is_absolute()) {
        throw std::invalid_argument("An absolute path is required; relative paths are not supported");
    }
    return path.lexically_normal();
}

std::string read_all(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to read file: " + to_utf8(path));
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void write_all(const fs::path& path, const std::string& content) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Unable to write file: " + to_utf8(path));
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!output) throw std::runtime_error("Failed while writing file: " + to_utf8(path));
}

std::vector<std::string> split_lines(const std::string& content) {
    std::vector<std::string> lines;
    std::istringstream input(content);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

size_t utf8_character_count(const std::string& text) {
    size_t count = 0;
    for (unsigned char ch : text) {
        if ((ch & 0xC0) != 0x80) ++count;
    }
    return count;
}

std::string utf8_prefix(const std::string& text, size_t characters) {
    size_t count = 0;
    size_t index = 0;
    while (index < text.size()) {
        if ((static_cast<unsigned char>(text[index]) & 0xC0) != 0x80) {
            if (count == characters) break;
            ++count;
        }
        ++index;
    }
    return text.substr(0, index);
}

bool regex_special(char ch) {
    const std::string specials = R"(.^$|()[]{}+\)";
    return specials.find(ch) != std::string::npos;
}

std::regex glob_regex(std::string pattern) {
    std::replace(pattern.begin(), pattern.end(), '\\', '/');
    std::string expression = "^";
    for (size_t i = 0; i < pattern.size(); ++i) {
        const char ch = pattern[i];
        if (ch == '*') {
            const bool double_star = i + 1 < pattern.size() && pattern[i + 1] == '*';
            if (double_star) {
                ++i;
                if (i + 1 < pattern.size() && pattern[i + 1] == '/') {
                    ++i;
                    expression += "(?:.*/)?";
                } else {
                    expression += ".*";
                }
            } else {
                expression += "[^/]*";
            }
        } else if (ch == '?') {
            expression += "[^/]";
        } else {
            if (regex_special(ch)) expression += '\\';
            expression += ch;
        }
    }
    expression += '$';
    return std::regex(expression, std::regex::ECMAScript);
}

struct GlobSpec {
    fs::path root;
    std::string relative_pattern;
};

GlobSpec parse_glob(const char* raw_pattern) {
    if (raw_pattern == nullptr) throw std::invalid_argument("glob pattern is null");
    fs::path pattern = from_utf8(raw_pattern);
    if (!pattern.is_absolute()) {
        throw std::invalid_argument(
            "An absolute glob pattern is required; relative globs are not supported");
    }

    fs::path root = pattern.root_path();
    fs::path relative_pattern;
    bool wildcard_found = false;
    for (const fs::path& component : pattern.relative_path()) {
        const std::string text = to_utf8(component);
        if (!wildcard_found && text.find_first_of("*?") == std::string::npos) {
            root /= component;
            continue;
        }
        wildcard_found = true;
        relative_pattern /= component;
    }

    if (!wildcard_found) {
        relative_pattern = root.filename();
        root = root.parent_path();
    }
    return {root.lexically_normal(), to_utf8(relative_pattern)};
}

std::vector<fs::path> matching_files(const char* raw_pattern) {
    GlobSpec spec = parse_glob(raw_pattern);
    const std::regex matcher = glob_regex(spec.relative_pattern);
    std::vector<fs::path> files;
    std::error_code error;
    fs::recursive_directory_iterator iterator(
        spec.root, fs::directory_options::skip_permission_denied, error);
    fs::recursive_directory_iterator end;
    for (; iterator != end; iterator.increment(error)) {
        if (error) {
            error.clear();
            continue;
        }
        if (!iterator->is_regular_file(error) || error) {
            error.clear();
            continue;
        }
        const fs::path absolute = iterator->path().lexically_normal();
        std::string relative = to_utf8(absolute.lexically_relative(spec.root));
        std::replace(relative.begin(), relative.end(), '\\', '/');
        if (std::regex_match(relative, matcher)) files.push_back(absolute);
    }
    std::sort(files.begin(), files.end(), [](const fs::path& left, const fs::path& right) {
        return to_utf8(left) < to_utf8(right);
    });
    return files;
}

std::string local_time_json() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    std::tm utc{};
#ifdef _WIN32
    localtime_s(&local, &timestamp);
    gmtime_s(&utc, &timestamp);
    const std::time_t local_as_utc = _mkgmtime(&local);
#else
    localtime_r(&timestamp, &local);
    gmtime_r(&timestamp, &utc);
    const std::time_t local_as_utc = timegm(&local);
#endif
    const long offset_seconds = static_cast<long>(local_as_utc - timestamp);
    const char sign = offset_seconds < 0 ? '-' : '+';
    const long absolute_offset = std::labs(offset_seconds);
    const long offset_hours = absolute_offset / 3600;
    const long offset_minutes = (absolute_offset % 3600) / 60;

    char date[16]{};
    char clock[16]{};
    std::strftime(date, sizeof(date), "%Y-%m-%d", &local);
    std::strftime(clock, sizeof(clock), "%H:%M:%S", &local);
#ifdef _WIN32
    // strftime("%Z") uses the active Windows code page rather than UTF-8.
    // Keep the ABI strictly UTF-8 by using a stable label on Windows.
    const std::string zone = "Local";
#else
    char zone_buffer[64]{};
    std::strftime(zone_buffer, sizeof(zone_buffer), "%Z", &local);
    const std::string zone(zone_buffer);
#endif
    std::ostringstream offset;
    offset << sign << std::setw(2) << std::setfill('0') << offset_hours
           << std::setw(2) << offset_minutes;
    const std::string compact_offset = offset.str();
    const std::string iso_offset = compact_offset.substr(0, 3) + ":" + compact_offset.substr(3);
    return "{\"iso\":" + quoted(std::string(date) + "T" + clock + iso_offset) +
           ",\"date\":" + quoted(date) + ",\"time\":" + quoted(clock) +
           ",\"timezone\":" + quoted(zone) + ",\"utc_offset\":" +
           quoted(compact_offset) + ",\"unix_timestamp\":" +
           std::to_string(static_cast<long long>(timestamp)) + "}";
}

}  // namespace

int agent_get_local_time(char* output, size_t output_size, size_t* required_size) {
    return run_json(local_time_json, output, output_size, required_size);
}

int agent_read_file(const char* path, int offset, int limit, char* output,
                    size_t output_size, size_t* required_size) {
    return run_json([&]() {
        const fs::path resolved = absolute_input_path(path);
        const std::string display = to_utf8(resolved);
        if (!fs::is_regular_file(resolved)) {
            return error_json("File does not exist: " + display, display);
        }
        if (offset < 1 || limit < 1 || limit > 500) {
            return error_json("offset must be positive and limit must be between 1 and 500", display);
        }
        const auto lines = split_lines(read_all(resolved));
        const size_t begin = std::min(static_cast<size_t>(offset - 1), lines.size());
        const size_t end = std::min(begin + static_cast<size_t>(limit), lines.size());
        std::ostringstream numbered;
        for (size_t i = begin; i < end; ++i) {
            if (i > begin) numbered << '\n';
            numbered << (i + 1) << ": " << lines[i];
        }
        return "{\"path\":" + quoted(display) + ",\"content\":" +
               quoted(numbered.str()) + ",\"start_line\":" + std::to_string(offset) +
               ",\"returned_lines\":" + std::to_string(end - begin) +
               ",\"total_lines\":" + std::to_string(lines.size()) +
               ",\"truncated\":" + (end < lines.size() ? "true" : "false") + "}";
    }, output, output_size, required_size);
}

int agent_list_files(const char* pattern, int max_results, char* output,
                     size_t output_size, size_t* required_size) {
    return run_json([&]() {
        if (max_results < 1) return error_json("max_results must be positive");
        const auto files = matching_files(pattern);
        const size_t count = std::min(files.size(), static_cast<size_t>(max_results));
        std::string json = "{\"files\":[";
        for (size_t i = 0; i < count; ++i) {
            if (i) json += ',';
            json += quoted(to_utf8(files[i]));
        }
        json += "],\"count\":" + std::to_string(count) + ",\"truncated\":" +
                (files.size() > count ? "true" : "false") + "}";
        return json;
    }, output, output_size, required_size);
}

int agent_search_text(const char* pattern, const char* file_glob,
                      int case_sensitive, int max_results, char* output,
                      size_t output_size, size_t* required_size) {
    return run_json([&]() {
        if (pattern == nullptr || *pattern == '\0') return error_json("pattern is required");
        if (max_results < 1) return error_json("max_results must be positive");
        std::regex::flag_type flags = std::regex::ECMAScript;
        if (!case_sensitive) flags |= std::regex::icase;
        std::regex matcher;
        try {
            matcher = std::regex(pattern, flags);
        } catch (const std::regex_error& exc) {
            return error_json(std::string("Invalid regular expression: ") + exc.what());
        }
        std::string json = "{\"matches\":[";
        int count = 0;
        bool truncated = false;
        for (const fs::path& path : matching_files(file_glob)) {
            std::vector<std::string> lines;
            try {
                lines = split_lines(read_all(path));
            } catch (...) {
                continue;
            }
            for (size_t index = 0; index < lines.size(); ++index) {
                if (!std::regex_search(lines[index], matcher)) continue;
                if (count >= max_results) {
                    truncated = true;
                    break;
                }
                if (count) json += ',';
                json += "{\"path\":" + quoted(to_utf8(path)) + ",\"line\":" +
                        std::to_string(index + 1) + ",\"text\":" +
                        quoted(utf8_prefix(lines[index], 500)) + "}";
                ++count;
            }
            if (truncated) break;
        }
        json += "],\"truncated\":" + std::string(truncated ? "true" : "false") + "}";
        return json;
    }, output, output_size, required_size);
}

int agent_write_file(const char* path, const char* content, char* output,
                     size_t output_size, size_t* required_size) {
    return run_json([&]() {
        if (content == nullptr) return error_json("content is null");
        const fs::path resolved = absolute_input_path(path);
        const std::string display = to_utf8(resolved);
        const bool existed = fs::exists(resolved);
        if (existed && !fs::is_regular_file(resolved)) {
            return error_json("Path is not a file: " + display, display);
        }
        if (!resolved.parent_path().empty()) fs::create_directories(resolved.parent_path());
        const std::string text(content);
        write_all(resolved, text);
        return "{\"path\":" + quoted(display) + ",\"operation\":" +
               quoted(existed ? "updated" : "created") +
               ",\"characters_written\":" + std::to_string(utf8_character_count(text)) + "}";
    }, output, output_size, required_size);
}

int agent_edit_file(const char* path, const char* old_string,
                    const char* new_string, int replace_all, char* output,
                    size_t output_size, size_t* required_size) {
    return run_json([&]() {
        if (old_string == nullptr || new_string == nullptr) {
            return error_json("replacement strings cannot be null");
        }
        const fs::path resolved = absolute_input_path(path);
        const std::string display = to_utf8(resolved);
        if (!fs::is_regular_file(resolved)) {
            return error_json("File does not exist: " + display, display);
        }
        const std::string old_value(old_string);
        const std::string new_value(new_string);
        if (old_value.empty() || old_value == new_value) {
            return error_json("old_string must be non-empty and different from new_string", display);
        }
        std::string content = read_all(resolved);
        size_t count = 0;
        for (size_t position = 0; (position = content.find(old_value, position)) != std::string::npos;
             position += old_value.size()) {
            ++count;
        }
        if (count == 0) return error_json("old_string was not found in the file", display);
        if (count > 1 && !replace_all) {
            return error_json("old_string occurs " + std::to_string(count) +
                              " times; provide more context or set replace_all to true", display);
        }
        const size_t replacements = replace_all ? count : 1;
        size_t position = 0;
        for (size_t done = 0; done < replacements; ++done) {
            position = content.find(old_value, position);
            content.replace(position, old_value.size(), new_value);
            position += new_value.size();
        }
        write_all(resolved, content);
        return "{\"path\":" + quoted(display) + ",\"replacements\":" +
               std::to_string(replacements) + "}";
    }, output, output_size, required_size);
}

int agent_start_process(const char* executable, const char* script,
                        const char* working_directory, char* output,
                        size_t output_size, size_t* required_size) {
    return run_json([&]() {
#ifdef _WIN32
        const fs::path executable_path = absolute_input_path(executable);
        const fs::path script_path = absolute_input_path(script);
        const fs::path working_path = absolute_input_path(working_directory);
        if (!fs::is_regular_file(executable_path)) {
            return error_json("Python executable does not exist: " + to_utf8(executable_path));
        }
        if (!fs::is_regular_file(script_path)) {
            return error_json("Flask entry point does not exist: " + to_utf8(script_path));
        }
        if (!fs::is_directory(working_path)) {
            return error_json("Working directory does not exist: " + to_utf8(working_path));
        }
        std::wstring command = L"\"" + executable_path.wstring() + L"\" \"" +
                               script_path.wstring() + L"\"";
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION process{};
        const BOOL started = CreateProcessW(
            executable_path.c_str(), command.data(), nullptr, nullptr, TRUE,
            CREATE_UNICODE_ENVIRONMENT, nullptr, working_path.c_str(), &startup,
            &process);
        if (!started) {
            return error_json("Unable to start Flask child process (Windows error " +
                              std::to_string(GetLastError()) + ")");
        }
        const DWORD process_id = process.dwProcessId;
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return "{\"process_id\":" + std::to_string(process_id) +
               ",\"executable\":" + quoted(to_utf8(executable_path)) +
               ",\"script\":" + quoted(to_utf8(script_path)) + "}";
#else
        (void)executable;
        (void)script;
        (void)working_directory;
        return error_json("agent_start_process is currently implemented only on Windows");
#endif
    }, output, output_size, required_size);
}
