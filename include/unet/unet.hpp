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

#ifndef __H__UNET__
#define __H__UNET__

#include <cstdarg>

namespace unet {

    enum LogLevel {DEBUG, INFO, ERROR, FATAL};

    typedef void (*UserLoggingFunction)(LogLevel,const char* fmt, va_list args);

    // Set a new default logging function, and return a pointer to the previous
    // one, if defined.
    UserLoggingFunction SetCustomLoggingFunction(UserLoggingFunction f); 
    // Print the version
    void LogPrintVersion(void);

    // Activation functions modify the result of each layer.
    typedef void (*DoubleActivationFunction)(double *,double *);
    typedef void (*FloatActivationFunction)(float *, float *);
    const int FunctionIsVectorised = 0x1;
    const int FunctionRequiresGPU  = 0x2; 

    // Error functions compare the output of the final
    // layer with a reference.
    typedef double (*DoubleErrorFunction)(double *);
    typedef float (*FloatErrorFunction)(float *); 

    // Returns true if OpenCL is supported on this system
    bool OpenCLSupported(void);

    // Function for testing
    void PropagateXOR(double *input, double *out);
}

#endif
