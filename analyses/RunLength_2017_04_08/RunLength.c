// RunLength.c
// RUNLENGTH - Run-length en- and decoding
// Run-length encoding splits a vector into one vector, which contains the
// elements without neighboring repetitions, and a second vector, which
// contains the number of repetitions. This can reduce the memory for storing
// the data or allow to analyse sequences.
//
// Encoding: [B, N, BI] = RunLength(X)
// Decoding: X          = RunLength(B, N)
// INPUT / OUTPUT:
//   X:  Full input signal, row or column vector.
//       Types: (U)INT8/16/32/64, SINGLE, DOUBLE, LOGICAL, CHAR.
//   B:  Compressed data, neighboring elements with the same value are removed.
//       B and X have the same types.
//   N:  Number of repetitions of the elements of B in X as DOUBLE or UINT8 row
//       vector.
//   BI: Indices of elements in B in X as DOUBLE row vector.
//
// RunLength(X, 'byte') replies N as UINT8 vector. Sequences with more than 255
// elements are split into parts. This needs less memory and processing time.
//
// NOTES:
//   The comparison treats NaNs as equal.
//   You can find a lot of RLE tools in the FileExchange already. This C-Mex is
//   about 5 times faster than good vectorized M-versions.
//   The M-file RunLength_M contains vectorized and loop M-code for education.
//
// EXAMPLES:
// Encode and decode:
//   [b, n] = RunLength([8, 9, 9, 10, 10, 10, 11])
//   x      = RunLength(b, n)
//   % b = [8,9,10,11],  n = [1,2,3,1],  x = [8,9,9,10,10,10,11]
// Limit counter to 255:
//   [b, n] = RunLength(ones(1, 257), 'byte')
//   % b = [1, 1],  n = uint8([255, 2])
// LOGICAL input:
//   [b, n] = RunLength([true(257, 1); false])
//   % b = [true; false],  n = [257, 1]
// Find the longest sequence:
//   x          = floor(rand(1, 1e6) * 2);
//   [b, n, bi] = RunLength(x);
//   [longestRun, index] = max(n);
//   longestPos = bi(index);
//
// COMPILATION:
// Alternative methods to get the compiled C-Mex file:
//   Implicit:      Running the M-file RunLength starts a compilation.
//   Installer:     InstallMex('RunLength.c', 'uTest_RunLength')
//   Manual Win:    mex -O RunLength.c
//   Manual Linux:  mex -O CFLAGS="\$CFLAGS -std=c99" RunLength.c
//   Download:      http://www.n-simon.de/mex
// Run the unit-test uTest_RunLength to test validity and speed!
//
// Tested: Matlab 6.5, 7.7, 7.8, 7.13, WinXP/32, Win7/64
//         Compiler: LCC3.8, BCC5.5, OWC1.8, MSVC2008/2010
// Does not compile under LCC2.4 shipped with Matlab6.5!
// Assumed Compatibility: higher Matlab versions, Linux, MacOS.
//
// Author: Jan Simon, Heidelberg, (C) 2013-2016 matlab.2010(a)n(MINUS)simon.de

/*
% $JRev: R-y V:025 Sum:VEM0ds3UL5N5 Date:08-Apr-2017 18:40:53 $
% $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
% $UnitTest: uTest_RunLength $
% $File: Tools\Mex\Source\RunLength.c $
% History:
% 001: 21-Feb-2013 00:39, First version.
% 006: 14-Mar-2013 08:21, Output has same column/row shape as input.
% 009: 29-Mar-2013 23:36, 3rd output BI for encoding.
% 025: 08-Apr-2017 18:32, 2nd output is a column now if 1st is one.
*/

// Includes and compiler specific settings: ------------------------------------
#include "mex.h"
#include <math.h>

// Error messages do not contain the function name in Matlab 6.5! This is not
// necessary in Matlab 7, but it does not bother:
#define ERR_HEAD "*** RunLength[mex]: "
#define ERR_ID   "JSimon:RunLength:"
#define ERROR(id,msg) mexErrMsgIdAndTxt(ERR_ID id, ERR_HEAD msg);

// Assume 32 bit addressing for Matlab 6.5:
// See MEX option "compatibleArrayDims" for MEX in Matlab >= 7.7.
#ifndef MWSIZE_MAX
#define mwSize  int32_T           // Defined in tmwtypes.h
#define mwIndex int32_T
#define MWSIZE_MAX MAX_int32_T
#endif

// LCC 2.4 (shipped with Matlab) and 3.8 (from the net) cannot compile int64_T
// as defined in mwtypes.h. After defining it here as "__int64" it works for
// v3.8:
#if defined(__LCC__)
#  if defined(MATLABVER) && MATLABVER == 605
#     error Cannot be compiled by LCC shipped with Matlab6.5
#  endif
   typedef __int64 int64_T;
#endif

// Prototypes: -----------------------------------------------------------------
mxArray *CreateOutput_decode(void *n, mwSize nb, mxClassID nClass,
                             mxClassID outClass, bool isColumn);
void CreateIndex(void *n, bool doByte, mwSize nb, double *index);

// Include subroutines for different data sizes: -------------------------------
// x and b are casted to integer types, such that NaN's and Inf's are treated
// as normal values and repeated NaN's are handled as a run.

#define DATA_TYPE int8_T    // For (U)INT8 and LOGICAL
#define FUNC_NAME(Fcn) Fcn ## _1Byte
#include "RunLength.inc"
#undef  DATA_TYPE
#undef  FUNC_NAME

#define DATA_TYPE int16_T   // For (U)INT16 and CHAR
#define FUNC_NAME(Fcn) Fcn ## _2Byte
#include "RunLength.inc"
#undef  DATA_TYPE
#undef  FUNC_NAME

#define DATA_TYPE int32_T   // For (U)INT32 and SINGLE:
#define FUNC_NAME(Fcn) Fcn ## _4Byte
#include "RunLength.inc"
#undef  DATA_TYPE
#undef  FUNC_NAME

#define DATA_TYPE int64_T   // For (U)INT64 and DOUBLE:
#define FUNC_NAME(Fcn) Fcn ## _8Byte
#include "RunLength.inc"
#undef  DATA_TYPE
#undef  FUNC_NAME

// Main function ===============================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwSize    nx, nb, n1, s1_Out1, s2_Out1;
  mxClassID classIn1, classIn2 = mxUNKNOWN_CLASS;
  void      *x, *b, *n;
  bool      doEncode, doByte, isColumn;
  uint8_T   *n_u;
  double    *n_d;
  int       ElementSize;
    
  // Check number of inputs, decide which action is performed: -----------------
  if (nrhs == 1) {             // 1 RHS: Encode
     doEncode = true;
     doByte   = false;
  } else if (nrhs == 2) {
     classIn2 = mxGetClassID(prhs[1]);
     switch (classIn2) {
        case mxDOUBLE_CLASS:   // 2 RHS, 2nd is a double: Decode
           doEncode = false;
           doByte   = false;
           break;
        case mxUINT8_CLASS:    // 2 RHS, 2nd is an UINT8: Decode_U8
           doEncode = false;
           doByte   = true;
           break;
        case mxCHAR_CLASS:     // 2 RHS, 2nd is a string: Encode_U8
           doEncode = true;
           doByte   = true;
           break;
        default:
           ERROR("BadTypeInput2",
                 "2nd input must be a double, uint8 or string.");
     }
  } else {                     // nrhs is neither 1 nor 2:
     ERROR("BadNInput", "1 or 2 inputs allowed.");
  }
  
  // Check number of outputs: --------------------------------------------------
  if (doEncode) {
     if (nlhs > 3) {      // 1 to 3 outputs accepted, 2 is the standard:
        ERROR("BadNOutput", "Encoding: Only up to 3 outputs allowed.");
     }
  } else if (nlhs > 1) {  // 1 output for decoding:
     ERROR("BadNOutput", "Decoding: Only 1 output allowed.");
  }
  
  // Check type of 1st input: --------------------------------------------------
  classIn1 = mxGetClassID(prhs[0]);
  if (!mxIsNumeric(prhs[0]) && !mxIsLogical(prhs[0]) && !mxIsChar(prhs[0])) {
     ERROR("BadTypeInput1", "Data must be numerical, logical or char.");
  }
  ElementSize = mxGetElementSize(prhs[0]);
    
  // Care for empty input: -----------------------------------------------------
  if (mxIsEmpty(prhs[0])) {
     // The 1st output has the same type as the input:
     plhs[0] = mxCreateNumericMatrix(0, 0, classIn1, mxREAL);
     if (nlhs >= 2) {  // The 2nd output is either a DOUBLE or an UINT8:
        if (doByte) {
           plhs[1] = mxCreateNumericMatrix(0, 0, mxUINT8_CLASS,  mxREAL);
        } else {
           plhs[1] = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
        }
        
        if (nlhs == 3) {
           plhs[2] = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
        }
     }
     return;
  }
  
  // Check if input is a vector: -----------------------------------------------
  n1 = mxGetNumberOfElements(prhs[0]);    // Number of elements of 1st input
  if ((mxGetM(prhs[0]) != 1 && mxGetN(prhs[0]) != 1) ||
      mxGetNumberOfDimensions(prhs[0]) != 2) {
     ERROR("BadSizeInput1", "Data must be a row or column vector.");
  }
  
  // Reply column vectors if 1st input is a column vector or a scalar:
  isColumn = (bool) (mxGetM(prhs[0]) > mxGetN(prhs[0]));
  
  // Get inputs, create output: ------------------------------------------------
  if (doEncode) {                         // [x]->[b,n]
     // Get input:
     x  = mxGetData(prhs[0]);
     nx = n1;
     
     // Create output - pre-allocate the maximum possible number of elements:
     plhs[0] = mxCreateNumericMatrix(n1, 1, classIn1, mxREAL);
     if (doByte) {
        plhs[1] = mxCreateNumericMatrix(nx, 1, mxUINT8_CLASS, mxREAL);
     } else {
        plhs[1] = mxCreateDoubleMatrix(nx, 1, mxREAL);
     }
     b = mxGetData(plhs[0]);
     n = mxGetData(plhs[1]);
     
     if (!plhs[0] || !plhs[1]) {    // Required in stand-alone mode only
        ERROR("MemoryExhausted", "Cannot create output.");
     }
     
  } else {  // !doEncode: B, N -> X,  Inputs and outputs for decoding:
     // Get input:
     b  = mxGetData(prhs[0]);
     nb = n1;
     n  = mxGetData(prhs[1]);
     
     if (mxGetNumberOfElements(prhs[1]) != nb) {
        ERROR("BadInputLength", "Both inputs must have the same size.");
     }
     
     // Create output with matching size:
     plhs[0] = CreateOutput_decode(n, nb, classIn2, classIn1, isColumn);
     x       = mxGetData(plhs[0]);
  }
  
  // Processing: ===============================================================
  if (doEncode)  {  // Encode [x]->[b,n]: --------------------------------------
     if (doByte) {  // Counter n as UINT8:
        // Encode with a limited run-length of 255 elements:
        n_u = (uint8_T *) n;
        switch (ElementSize) {
           case 8:   // DOUBLE, (U)INT64:
              Encode_U8_8Byte((int64_T *) x, nx, (int64_T *) b, n_u, &nb);
              break;
           case 4:   // SINGLE, (U)INT32:
              Encode_U8_4Byte((int32_T *) x, nx, (int32_T *) b, n_u, &nb);
              break;
           case 2:   // CHAR, (U)INT16:
              Encode_U8_2Byte((int16_T *) x, nx, (int16_T *) b, n_u, &nb);
              break;
           case 1:   // LOGICAL, (U)INT8:
              Encode_U8_1Byte((int8_T *)  x, nx, (int8_T *)  b, n_u, &nb);
              break;
           default:  // Other types have been excluded already actually:
              ERROR("BadTypeInput1", "Class of input not accepted.");
        }
        
     } else {        // Counter n as DOUBLE:
        n_d = (double *) n;
        switch (ElementSize) {
           case 8:   // DOUBLE, (U)INT64:
              Encode_8Byte((int64_T *) x, nx, (int64_T *) b, n_d, &nb);
              break;
           case 4:   // SINGLE, (U)INT32:
              Encode_4Byte((int32_T *) x, nx, (int32_T *) b, n_d, &nb);
              break;
           case 2:   // CHAR, (U)INT16:
              Encode_2Byte((int16_T *) x, nx, (int16_T *) b, n_d, &nb);
              break;
           case 1:   // LOGICAL, (U)INT8:
              Encode_1Byte((int8_T *) x,  nx, (int8_T *) b,  n_d, &nb);
              break;
           default:  // Other types have been excluded already actually:
              ERROR("BadTypeInput1", "Class of input not accepted.");
        }
     }
     
     // Crop unneeded pre-allocated elements: ----------------------------------
     if (isColumn) {
        mxSetM(plhs[0], nb);
        mxSetM(plhs[1], nb);
     } else {
        mxSetN(plhs[0], nb);
        mxSetN(plhs[1], nb);
        mxSetM(plhs[0], (mwSize) 1);
        mxSetM(plhs[1], (mwSize) 1);
     }
     
     // mxRealloc is *not* mandatory and under some circumstances it even does
     // not free the memory in modern Matlab versions:
     if (nb != nx) {
        mxSetData(plhs[0], mxRealloc(b, nb * ElementSize));
        mxSetData(plhs[1], mxRealloc(n, nb * mxGetElementSize(plhs[1])));
     }
     
     // Create 3rd output as post-processing: ----------------------------------
     // Collecting the indices on the fly might be faster, but doing this
     // afterwards allowes to allocate the correct number of elements directly:
     if (nlhs == 3) {
        if (isColumn) {
           plhs[2] = mxCreateDoubleMatrix(nb, 1, mxREAL);
        } else {
           plhs[2] = mxCreateDoubleMatrix(1, nb, mxREAL);
        }
        CreateIndex(mxGetData(plhs[1]), doByte, nb, mxGetPr(plhs[2]));
     }
     
  } else {           // Decode [b,n]->[x]: -------------------------------------
     if (doByte) {   // Counter n as UINT8:
        n_u = (uint8_T *) n;
        switch (ElementSize) {
           case 8:   // DOUBLE, (U)INT64:
              Decode_U8_8Byte((int64_T *) b, n_u, nb, (int64_T *) x);
              break;
           case 4:   // SINGLE, (U)INT32:
              Decode_U8_4Byte((int32_T *) b, n_u, nb, (int32_T *) x);
              break;
           case 2:   // CHAR, (U)INT16:
              Decode_U8_2Byte((int16_T *) b, n_u, nb, (int16_T *) x);
              break;
           case 1:   // LOGICAL, (U)INT8:
              Decode_U8_1Byte((int8_T *) b,  n_u, nb, (int8_T *) x);
              break;
           default:  // Other types have been excluded already actually:
              ERROR("BadTypeInput1", "Class of input not accepted.");
        }
        
     } else {        // Counter n as DOUBLE:
        n_d = (double *) n;
        switch (ElementSize) {
           case 8:   // DOUBLE, (U)INT64:
              Decode_8Byte((int64_T *) b, n_d, nb, (int64_T *) x);
              break;
           case 4:   // SINGLE, (U)INT32:
              Decode_4Byte((int32_T *) b, n_d, nb, (int32_T *) x);
              break;
           case 2:   // CHAR, (U)INT16:
              Decode_2Byte((int16_T *) b, n_d, nb, (int16_T *) x);
              break;
           case 1:   // LOGICAL, (U)INT8:
              Decode_1Byte((int8_T *) b,  n_d, nb, (int8_T *) x);
              break;
           default:  // Other types have been excluded already actually:
              ERROR("BadTypeInput1", "Class of input not accepted.");
        }
     }
  }

  return;
}

// *****************************************************************************
mxArray *CreateOutput_decode(void *n, mwSize InLen, mxClassID nClass,
                             mxClassID OutClass, bool isColumn)
{
  // Count total number of output elements for decoding.
  // All elements must be >= 0, finite and the sum is checked to be smaller than
  // 2^52.
  
  mwSize  dim1, dim2;
  double  OutLen = 0.0, *n_d;
  uint8_T *n_i;
  mxArray *Out;
  
  // Count number of elements in the output:
  switch (nClass) {
     case mxDOUBLE_CLASS:
        n_d = (double *) n;
        while (InLen-- != 0) {   // Counter must be non-negative integers
           if (*n_d != floor(*n_d) || *n_d < 0) {
              ERROR("BadCounterValue", "Counter must be integer >= 0.");
           }
           OutLen += *n_d++;
        }
        break;
        
     case mxUINT8_CLASS:
        n_i = (uint8_T *) n;
        while (InLen-- != 0) {
           OutLen += (double) *n_i++;
        }
        break;
        
     default:
        ERROR("BadTypeInput2", "The 2nd input must be a DOUBLE or UINT8.");
  }
  
  if (!mxIsFinite(OutLen)) {          // Reject NaNs and Infs in the counter
     ERROR("InfiniteCounter", "Counter must have finite values.");
  }

  if (OutLen > 4503599627370496.0 ||   // Max representable integer: 2^52
      OutLen > (double) MWSIZE_MAX) {  // Max array size for 32/64 bits
     ERROR("CounterTooLarge", "Total output length exceeds accurate range.");
  }
  
  // Create output vector with same orientation as the input:
  if (isColumn) {
     dim1 = (mwSize) OutLen;
     dim2 = 1;
  } else {
     dim1 = 1;
     dim2 = (mwSize) OutLen;
  }
  Out = mxCreateNumericMatrix(dim1, dim2, OutClass, mxREAL);
  
  if (Out == NULL) {                  // Required in stand-alone mode only
     ERROR("MemoryExhausted", "Cannot create output.");
  }
  
  return Out;
}

// *****************************************************************************
void CreateIndex(void *n, bool doByte, mwSize nb, double *index)
{
  // Create index vector of B in X for encoding.
  // Overflow or Inf/NaN are not possible, because the run lengths n have been
  // calculated here before.
  // This function is called for nb >= 1 only, such that [index] is not empty.
  // Equivalent M-code: index = cumsum([1, n(1:end-1)])
  
  uint8_T *n_u;
  double  *n_d, c = 1.0;
  
  *index++ = c;
     
  if (doByte) {  // Counter n as UINT8:
     n_u = (uint8_T *) n;
     while (--nb) {
        c       += *n_u++;
        *index++ = c;
     }
     
  } else {       // Counter n as DOUBLE:
     n_d = (double *) n;
     while (--nb) {
        c       += *n_d++;
        *index++ = c;
     }
  }
  
  return;
}
