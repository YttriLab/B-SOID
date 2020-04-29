// RunLength.inc
// Support function for RunLength.c
//
// Tested: Matlab 6.5, 7.7, 7.8, 7.13, WinXP/32, Win7/64
//         Compiler: LCC3.8, BCC5.5, OWC1.8, MSVC2008/2010
// Assumed Compatibility: higher Matlab versions, Linux, MacOS.
//
// Author: Jan Simon, Heidelberg, (C) 2010-2016 matlab.2010(a)n(MINUS)simon.de

/*
% $JRev: R-g V:006 Sum:Pe4tVBc8BZUi Date:25-Apr-2016 00:54:28 $
% $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
% $File: Tools\Mex\Source\RunLength.inc $
% History:
% 001: 03-Mar-2013 00:39, First version.
*/

// Prototypes:
void FUNC_NAME(Encode)    (DATA_TYPE *x, mwSize nx,
                           DATA_TYPE *b, double  *n, mwSize *nb);
void FUNC_NAME(Encode_U8) (DATA_TYPE *x, mwSize nx,
                           DATA_TYPE *b, uint8_T *n, mwSize *nb);
void FUNC_NAME(Decode)    (DATA_TYPE *b, double  *n, mwSize nb, DATA_TYPE *x);
void FUNC_NAME(Decode_U8) (DATA_TYPE *b, uint8_T *n, mwSize nb, DATA_TYPE *x);


// *****************************************************************************
// **                                 ENCODING                                **
// *****************************************************************************
void FUNC_NAME(Encode) (DATA_TYPE *x, mwSize nx,
                        DATA_TYPE *b, double *n, mwSize *nb)
{
  // INPUT:
  //   x:  Original data stream with repeated neighboring elements.
  //   nx: Number of elements in x.
  // OUTPUT:
  //   b:  Data without repeated neighboring elements.
  //       Pre-allocated to nx elements
  //   n:  Run-length for each element of b.
  //   nb: Length of b and n.
  //
  // x and b are casted to integer types, such that NaN's and Inf's are treated
  // as normal values and repeated NaN's are handled as a run.

  DATA_TYPE *xi, *xf, *b0 = b;
    
  xf = x + nx;              // Final element of input x
  xi = x;                   // First value
  while (++x < xf) {        // Loop from 2nd element of x to end
     if (*x != *xi) {       // Value of x has changed
        *n++ = (double) (x - xi);  // Number of repetitions
        *b++ = *xi;         // Store former value of x
        xi   = x;           // Remember new value of x
     }
  }
  *n = (double) (x - xi);   // Flush last element
  *b = *xi;
  
  *nb = b - b0 + 1;         // Reply number of found runs for re-allocation
}

// *****************************************************************************
// **                                 DECODING                                **
// *****************************************************************************
void FUNC_NAME(Decode)(DATA_TYPE *b, double *n, mwSize nb, DATA_TYPE *x)
{
  // INPUT:
  //   b:  Pointer to the vector of values.
  //   n:  Pointer to vector of counters.
  //   nb: Length of input vectors b and n.
  // OUTPUT:
  //   x:  Vector of inflated output values. This is allocated in the caller.
  
  mwSize ni;
  
  while (nb-- != 0) {     // Inflate the data
     ni = (mwSize) *n++;  // Number of repetitions
     while (ni-- != 0) {  // Fill output x with repeated values
        *x++ = *b;
     }
     b++;                 // Next element of b
  }
}

// *****************************************************************************
// **                 ENCODING WITH LIMITED RUN-LENGTH                        **
// *****************************************************************************
void FUNC_NAME(Encode_U8)(DATA_TYPE *x, mwSize nx,
                          DATA_TYPE *b, uint8_T *n, mwSize *nb)
{
  // Same method as for [n] of type double, but the run-length is limited to 255
  // and stored compacter in a UINT8 vector.
  
  DATA_TYPE *xi, *xf, *b0 = b;
  mwSize d;

  xf = x + nx;               // Final element of x
  xi = x;                    // First value of x
  while (++x < xf) {         // Loop from 2nd to last element of x
     if (*x != *xi) {        // If value has changed
        d = x - xi;          // Number of elements until last change
        while (d > 255UL) {  // Length exceeds 255
           *n++ = (uint8_T) 255;
           *b++ = *xi;
           d   -= 255UL;     // Reduce length
        }
        *n++ = (uint8_T) d;  // Last chunk has <= 255 elements
        *b++ = *xi;          // Store value of x and proceed to next b
        xi   = x;            // Remember new value of x
     }
  }
  
  d = x - xi;                // Flush last element, same as inside the loop
  while (d > 255UL) {
     *n++ = (uint8_T) 255;
     *b++ = *xi;
     d   -= 255UL;
  }
  *n = (uint8_T) (d);
  *b = *xi;

  *nb = b - b0 + 1;          // Reply number of found blocks
}

// *****************************************************************************
// **                 DECODING WITH LIMITED RUN-LENGTH                        **
// *****************************************************************************
void FUNC_NAME(Decode_U8)(DATA_TYPE *b, uint8_T *n, mwSize nb, DATA_TYPE *x)
{
  // Same method as for standard decoding, but [n] is an UINT8.
  // The body of the function does not depend on the DATA_TYPE.
    
  uint8_T ni;
  
  while (nb-- != 0) {     // Loop over all elements of b
     ni = *n++;           // Number of repetitions
     while (ni-- != 0) {  // Write current b to output
        *x++ = *b;
     }
     b++;                 // Next element of b
  }
}
