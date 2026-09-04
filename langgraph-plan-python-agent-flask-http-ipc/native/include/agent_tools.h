#pragma once

#include <stddef.h>

#ifdef _WIN32
#define AGENT_TOOLS_API extern "C" __declspec(dllexport)
#else
#define AGENT_TOOLS_API extern "C" __attribute__((visibility("default")))
#endif

// Every function returns a UTF-8 JSON object through a caller-owned buffer.
// Return 0 on success, 1 when the buffer is missing/too small, and -1 for an
// invalid ABI argument. required_size includes the trailing NUL byte.
AGENT_TOOLS_API int agent_get_local_time(
    char* output, size_t output_size, size_t* required_size);

AGENT_TOOLS_API int agent_read_file(
    const char* path, int offset, int limit,
    char* output, size_t output_size, size_t* required_size);

AGENT_TOOLS_API int agent_list_files(
    const char* pattern, int max_results,
    char* output, size_t output_size, size_t* required_size);

AGENT_TOOLS_API int agent_search_text(
    const char* pattern, const char* file_glob, int case_sensitive,
    int max_results, char* output, size_t output_size, size_t* required_size);

AGENT_TOOLS_API int agent_write_file(
    const char* path, const char* content,
    char* output, size_t output_size, size_t* required_size);

AGENT_TOOLS_API int agent_edit_file(
    const char* path, const char* old_string, const char* new_string,
    int replace_all, char* output, size_t output_size, size_t* required_size);

// Start the Flask child with the current process environment. Executable and
// script must be absolute. An empty working_directory makes the child inherit
// the parent process directory; otherwise it must be an absolute directory.
AGENT_TOOLS_API int agent_start_process(
    const char* executable, const char* script, const char* working_directory,
    char* output, size_t output_size, size_t* required_size);
