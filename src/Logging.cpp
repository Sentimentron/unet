/* 
 * Copyright (c) 2015, Richard Townsend and Contributors. All rights reserved. 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdarg>
#include <pthread.h>
#include <stdio.h>

#include <unet/unet.hpp>

using namespace unet;

static UserLoggingFunction userLogFunction;
static pthread_mutex_t uNetConfigurationLock = PTHREAD_MUTEX_INITIALIZER;

/// Set a new default logging function and return a pointer to the
/// previous one.
UserLoggingFunction SetCustomLoggingFunction(UserLoggingFunction f) {
    UserLoggingFunction ret = nullptr;
    pthread_mutex_lock(&uNetConfigurationLock);
    ret = userLogFunction;
    userLogFunction = f;
    pthread_mutex_unlock(&uNetConfigurationLock);
    return ret;

}

static const char *LogLevelToString(LogLevel l) {
    switch(l) {
        case DEBUG:
            return "DEBUG";
        case INFO:
            return "INFO";
        case ERROR:
            return "ERROR";
        case FATAL:
            return "FATAL";
        default:
            return "UNKNOWN";
    }
}

void Log(LogLevel l, const char *fmt...) {

    va_list args;
    va_start(args, fmt);

    if (userLogFunction != nullptr) {
        userLogFunction(l, fmt, args);
    } else {
        fprintf(stderr, "μNet/%s", LogLevelToString(l));
        vfprintf(stderr, fmt, args);
    }

    va_end(args);
}
