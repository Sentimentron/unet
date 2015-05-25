#ifndef _H_UNET_INTERNAL

#include <unet/unet.hpp>

namespace unet {
void Log(LogLevel l, const char *fmt...);
void AssertFailure(const char *, const char *, int);

#define UASSERT(cond) \
    do { \
        AssertFailure(#cond, __FILE__, __LINE__); \
    } while (0)
}

#endif
