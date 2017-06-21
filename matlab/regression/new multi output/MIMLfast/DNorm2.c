// DNorm2.c
// Fast euclidian norm over N.th dimension of a DOUBLE array
// Y = DNorm2(X, N)
// INPUT:
//   X: Real DOUBLE array.
//   N: Dimension to operate on.
// OUTPUT:
//   Y: Euclidian norm over N.th dimension: Y = sqrt(sum(X .* X, N)).
//      Length of N.th dimension of Y is 1.
//      NaN's are considered.
//
// Matlab's built-in function NORM is fast for vectors, but for matrices the
// matrix norm is replied. Other efficient methods for vectors(!):
//   Y = sqrt(sum(X .* X));
//   Y = sqrt(X * X');       % row vectors, faster than DNorm2!
// And James Tursa's MTIMESX is very fast for vectors also and can operate on
// the 1st dimension of arrays also:
//   X = rand(100, 100);  X = reshape(X, 100, 1, 100);
//   Y = sqrt(mtimesx(X, 't', X));
// But for arrays DNorm2 is faster, and I do not see a way to apply MTIMESX for
// trailing dimensions without time-consuming transpositions.
//
// COMPILATION:
//   This function must be compiled before using:
//     mex -O DNorm2.c
//   Linux: consider C99 comments:
//     mex -O CFLAGS="\$CFLAGS -std=C99" DNorm2.c
//   Pre-compiled Mex: http://www.n-simon.de/mex
//
// TEST: Run uTest_DNorm2 to test validity and speed.
//
// NOTES:
// Different algorithms are called for special cases to increase the speed:
//   1. SIZE(X, N) == 1: The norm is a simple ABS.
//   2. [A x 3] and [3 x B] matrices: Unrolled loops.
//   3. N is the first singelton dimension: Fastest, because neighboring
//      elements are accessed.
//   4. A rowwise and a columnwise calculations:
//      ROWWISE: Calculte the norm over a subvector and copy the result to an
//               element of the output. The output is accessed in contiguous
//               memory, but the input is not. Fast for [long x short] arrays.
//      COLUMNWISE: Accumulate the squared elements of the input in a chunk
//               of the output at first. In the last iteration calculate the
//               SQRT in addition. Input and output are accessed in contiguous
//               memory, but multiple iterations over the output elements are
//               necessary. Fast for [short x long] arrays.
//      I've spent some evenings to find out an optimal strategy to define
//      "long" and "short", but I got only a rough solution:
//        L  = size(X, N)
//        nX = numel(X)
//        if     L <= 7         >> ROW
//        elseif L <= 13
//           if nX / L < 10000  >> COLUMN
//           else               >> ROW
//        else
//           if nX / L < 75000  >> COLUMN
//           else               >> ROW
//
//      This does not look smart, but it catchs at least the extremal shapes.
//      There is a sharp limit between the 10000 and the 75000, which seems to
//      depend on the size of the processor cache.
//      ** Please feel free to find better limits matching your processor! **
//   5. With the MSVC++2008 compiler the /arch:SSE2 and /fp:fast flags are very
//      important: The SSE2 flag sometimes increases and sometimes decreases the
//      speed by a factor of 2.5, so I decided to disable SSE2 instruction.
//      The fp:fast flag produces faster code in all of my tests. Compared to
//      fp:precise the results differ in the magnitude of EPS
//      (not EPS*numel(X)!). Therefore I decided to simulate /fp:fast by
//      disabling fp:precise and fp:except and enabling fp:contract by #pragma's
//      if the MSVC compiler is used.
//      ** Please feel free to find better compiler options! **
//   6. I cannot test OMP mutlithreading on my currently used single-core
//      machine. A mutli-threaded version of DNorm2 will be *much* faster!
//      ** Please feel free to assist me with multi-threading! **
//
// Tested: Matlab 6.5, 7.7, 7.8, WinXP, 32bit
//         Compiler: LCC2.4/3.8, BCC5.5, OWC1.8, MSVC2008
// Assumed Compatibility: higher Matlab versions, Mac, Linux, 64bit
// Author: Jan Simon, Heidelberg, (C) 2006-2010 matlab.THISYEAR(a)nMINUSsimon.de

/*
% $JRev: R2.00C V:043 Sum:d+sMZTKCxTs9 Date:16-Oct-2010 00:11:26 $
% $License: BSD $
% $File: Tools\Mex\Source\DNorm2.c $
% History:
% 013: Extra treatment of [T x 3] array, because they appear often
%      in MoMo.
%      Unfortunately the Matlab LCC-Compiler produces faster code,
%      if one unused variable is added...
% 022: 08-Aug-2006 15:40, Renamed: SqrtSumQuad -> DNorm2.
%      BCC compiler accepts SQRT(NaN) on included fastmath.h without an
%      exception. With _control87 the precision is set to 64 bits.
% 034: 04-Sep-2009 00:41, 4% faster with sum stored in Y directly.
% 038: 25-Sep-2010 22:58, 64bit, 1st non-singelton as default (was 1st).
% 043: 16-Oct-2010 00:08, Rounding the dim failed for 64 bit.
*/

#include "mex.h"
#include <stdlib.h>
#include <float.h>
#include <string.h>

#ifdef __BORLANDC__     // SQRT without exceptions for NaN
#include <fastmath.h>
#else
#include <math.h>
#endif

// Assume 32 bit addressing for Matlab 6.5:
// See MEX option "compatibleArrayDims" for MEX in Matlab >= 7.7.
#ifndef MWSIZE_MAX
#define mwSize  int32_T           // Defined in tmwtypes.h
#define mwIndex int32_T
#define MWSIZE_MAX MAX_int32_T
#endif

// Disable the /fp:precise flag to increase the speed on MSVC compiler:
#ifdef _MSC_VER
#pragma float_control(except, off)    // disable exception semantics
#pragma float_control(precise, off)   // disable precise semantics
#pragma fp_contract(on)               // enable contractions
// #pragma fenv_access(off)              // disable fpu environment sensitivity
#endif

// Error messages do not contain the function name in Matlab 6.5! This is not
// necessary in Matlab 7, but it does not bother:
#define ERR_HEAD "*** DNorm2[mex]: "
#define ERR_ID   "JSimon:DNorm2:"

// Prototypes
void CalcTx3(double *X, const mwSize M, double *Yp);
void Calc3xT(double *X, const mwSize nX, double *Yp);
void CalcDim1(double *X, const mwSize M, const mwSize nDX, double *Yp);
void CalcCol(double *Xp, const mwSize step, const mwSize nX, const mwSize nDX,
             double *Yp);
void CalcRow(double *Xp, const mwSize step, const mwSize nX, const mwSize nDX,
             double *Yp);
void CalcAbs(double *Xp, const mwSize nX, double *Yp);
     
void C87to64bit(int Cmd);

mwSize FirstNonSingeltonDim(const mwSize Xndim, const mwSize *Xdim);
mwSize GetStep(const mwSize *Xdim, const mwSize N);

// Main function ===============================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double       *Xp, *Yp, Nd;
  mwSize       nX, nDX, ndimX, *dimY, N, Step;
  const mwSize *dimX;
  
  // Proper number of arguments:
  if (nrhs != 1 && nrhs != 2) {
     mexErrMsgIdAndTxt(ERR_ID   "BadNInput",
                       ERR_HEAD "1 or 2 inputs required.");
  }
  if (nlhs > 1) {
     mexErrMsgIdAndTxt(ERR_ID   "BadNOutput",
                       ERR_HEAD "1 output allowed.");
  }
  
  if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
     mexErrMsgIdAndTxt(ERR_ID   "BadTypeInput1",
                       ERR_HEAD "Input must be real double.");
  }
  
  // Create input matrix:
  Xp    = mxGetPr(prhs[0]);
  nX    = mxGetNumberOfElements(prhs[0]);
  ndimX = mxGetNumberOfDimensions(prhs[0]);
  dimX  = mxGetDimensions(prhs[0]);
  
  // Get dimension to operate on:
  if (nrhs == 1) {
     N    = FirstNonSingeltonDim(ndimX, dimX);
     Step = 1;
     
  } else if (mxIsNumeric(prhs[1]))  {  // 2nd input used:
     switch (mxGetNumberOfElements(prhs[1])) {
        case 0:  // Use 1st non-singelton dim if 2nd input is []:
           N    = FirstNonSingeltonDim(ndimX, dimX);
           Step = 1;
           break;
           
        case 1:  // Numerical scalar:
           Nd = mxGetScalar(prhs[1]) - 1;
           N  = (mwSize) Nd;
           if (Nd < 0 || Nd != floor(Nd)) {
              mexErrMsgIdAndTxt(ERR_ID   "BadValueInput2",
                                ERR_HEAD
                                "Dimension must be a positive integer scalar.");
           }
           
           if (N >= ndimX) {
              // Treat imaginated trailing dimensions as singelton, as usual in
              // Matlab:
              plhs[0] = mxCreateNumericArray(ndimX, dimX,
                                             mxDOUBLE_CLASS, mxREAL);
              CalcAbs(Xp, nX, mxGetPr(plhs[0]));
              return;
              
           }
           
           Step = GetStep(dimX, N);
           break;
           
        default:
          mexErrMsgIdAndTxt(ERR_ID   "BadSizeInput2",
                            ERR_HEAD "2nd input must be scalar index.");
     }
     
  } else {  // 2nd input is not numeric:
     mexErrMsgIdAndTxt(ERR_ID   "BadTypeInput2",
                       ERR_HEAD "2nd input must be scalar index.");
  }
  
  // Return fast on empty input matrix
  if (nX == 0) {
     plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
     return;
  }
  
  // Create output matrix:
  nDX     = dimX[N];               // Number of elements to sum over
  dimY    = (mwSize *) mxMalloc(ndimX * sizeof(mwSize));  // Dimensions
  memcpy(dimY, dimX, ndimX * sizeof(mwSize));
  dimY[N] = 1;                     // Sum let one dimension collapse
  plhs[0] = mxCreateNumericArray(ndimX, dimY, mxDOUBLE_CLASS, mxREAL);
  Yp      = mxGetPr(plhs[0]);
  mxFree(dimY);
  
  // ABS is a cheap norm over singelton dimension:
  if (nDX == 1) {
     CalcAbs(Xp, nX, Yp);
     return;
  }
  
  C87to64bit(1);     // Enable 80bit registers on Intel-CPU
  
  if (Step == 1) {   // Easier algorithm for 1st dimension
     if (nDX == 3) { // Special treatment of common [3 x T] arrays:
        Calc3xT(Xp, nX, Yp);
     } else {
        CalcDim1(Xp, nX, nDX, Yp);
     }
     
  } else if (ndimX == 2 && nDX == 3) {  // Process 2nd dimension!
     // Special treatment of common [T x 3] arrays:
     CalcTx3(Xp, Step, Yp);
     
  } else {           //  >= 1, general algorithm
     // Column oriented method: fast for [short x large]
     // Row oriented method:    fast for [large x short]
     // But after a lot of tests, I got only a coarse idea of what "short" and
     // "large" could be. I assume, this depends on the sizes of the 1st and
     // 2nd level caches.
     
     if (nDX <= 7) {
        CalcRow(Xp, Step, nX, nDX, Yp);
        
     } else if (nDX <= 13) {
        if (nX / nDX < 10000) {
           CalcCol(Xp, Step, nX, nDX, Yp);
        } else {
           CalcRow(Xp, Step, nX, nDX, Yp);
        }
        
     } else {
        if (nX / nDX < 75000) {
           CalcCol(Xp, Step, nX, nDX, Yp);
        } else {
           CalcRow(Xp, Step, nX, nDX, Yp);
        }
     }
  }
  
  C87to64bit(0);     // Reset 80bit register usage
  
  return;
}

// Subroutines =================================================================
mwSize FirstNonSingeltonDim(const mwSize Xndim, const mwSize *Xdim)
{
  // Get first non-singelton dimension - zero based.
  mwSize N;
  
  for (N = 0; N < Xndim; N++) {
     if (Xdim[N] != 1) {
        return (N);
     }
  }
  
  return (0);  // Use the first dimension if all dims are 1
}

// =============================================================================
mwSize GetStep(const mwSize *Xdim, const mwSize N)
{
  // Get step size between elements of a subvector in the N'th dimension.
  // This is the product of the leading dimensions.
  const mwSize *XdimEnd, *XdimP;
  mwSize       Step;
  
  Step    = 1;
  XdimEnd = Xdim + N;
  for (XdimP = Xdim; XdimP < XdimEnd; Step *= *XdimP++) ; // empty loop
  
  return (Step);
}

// =============================================================================
void CalcDim1(double *Xp, const mwSize nX, const mwSize nDX, double *Yp)
{
  // Norm over first dimension with simple algorithm.
  // For the LCC compiler: Accumulating the Sum in *Yp directly is faster!
  // [10 x 10.000] matrix: 60% faster with /arch:SSE2 /fp:fast in MSVC 2008.
  // [100.000 x 1] matrix: 33% slower with /arch:SSE2 /fp:fast in MSVC 2008

  double *XEnd, *XDEnd, Sum;
  
  for (XEnd = Xp + nX; Xp < XEnd; ) {
     Sum = 0.0;
     for (XDEnd = Xp + nDX; Xp < XDEnd; Xp++) {
        Sum += *Xp * *Xp;
     }
     *Yp++ = sqrt(Sum);
  }
  
  return;
}

// =============================================================================
void Calc3xT(double *Xp, const mwSize nX, double *Yp)
{
  // Norm over first dimension of length 3.
  // This is just a few percent faster than the Dim1 method.
  
  mwSize i;
  
  for (i = 0; i < nX; i += 3) {
     *Yp++ = sqrt(Xp[i]*Xp[i] + Xp[i+1]*Xp[i+1] + Xp[i+2]*Xp[i+2]);
  }
  
  return;
}

// =============================================================================
void CalcTx3(double *X, const mwSize M, double *Yp)
{
  // Unrolled loops for [Mx3] arrays.
  
  double *X1, *X2, *X3, *XEnd;
  X1 = X;
  X2 = X1 + M;
  X3 = X2 + M;
  
  for (XEnd = X1 + M; X1 < XEnd; X1++, X2++, X3++) {
     *Yp++ = sqrt(*X1 * *X1 + *X2 * *X2 + *X3 * *X3);
  }
  
  return;
}

// =============================================================================
void CalcCol(double *Xp, const mwSize step, const mwSize nX, const mwSize nDX,
             double *Yp)
{
  // General case: Process any dimension of a multi-dimensional array.
  // Column oriented approach: Process contiguous memory blocks of input and
  // output, what needs multiple loops over the output.
  // Fast for: [short x large] array.
  
  mwSize i, j;
  double *Yq, *Xq, *Xf;
  
  // Loop over specified dimension:
  Yq = Yp;
  Xq = Xp;
  Xf = Xq + nX;       // Final element of input
  while (Xq < Xf) {
     // First block: set
     for (j = 0; j < step; j++) {
        Yq[j] = Xq[j] * Xq[j];
     }
     Xq += step;      // Next input block
     
     // Middle blocks: add
     for (i = 2; i < nDX; i++) {   // Or for(i = 1; i < nDX-1; i++)
        for (j = 0; j < step; j++) {
           Yq[j] += Xq[j] * Xq[j];
        }
        Xq += step;   // Next input block
     }
     
     // Last block: add and sqrt
     for (j = 0; j < step; j++) {
        Yq[j] = sqrt(Yq[j] + Xq[j] * Xq[j]);
     }
     Xq += step;      // Next input block
     
     Yq += step;      // Next output block
  }
  
  return;
}

// =============================================================================
void CalcRow(double *Xp, const mwSize Step, const mwSize nX, const mwSize nDX,
             double *Yp)
{
  // General case: Process any dimension of a multi-dimensional array.
  // Row oriented approach: Process contiguous memory blocks of output, but
  // steps for the input.
  // Fast for: [large x short] array.
   
  double *Xq, *Xf, *XDf, *XSf, Sum;
  mwSize nDXstep;
  
  // Distance between first and last element of X in specified dim:
  nDXstep = nDX * Step;
  
  Xf = Xp + nX;
  while (Xp < Xf) {
     // Loop over specified dimension:
     for (XSf = Xp + Step; Xp < XSf; Xp++) {
        Sum = 0.0;
        XDf = Xp + nDXstep;
        for (Xq = Xp; Xq < XDf; Xq += Step) {
           Sum += *Xq * *Xq;
        }
        *Yp++ = sqrt(Sum);
     }
     
     // Move pointer to the next chunk:
     Xp += nDXstep - Step;
  }
  
  return;
}


// =============================================================================
void CalcAbs(double *Xp, const mwSize nX, double *Yp)
{
  // Simple norm over singelton dimension: ABS.
  double *Yf = Yp + nX;
  
  while (Yp < Yf) {
     if (*Xp >= 0) {
        *Yp++ = *Xp++;
     } else {
        *Yp++ = -*Xp++;
     }
  }

  return;
}
     
// =============================================================================
void C87to64bit(int Cmd)
{
  // Store original _control87 value and set to precision to 64 bits.
  // It seems as Matlab resets the floating point processor flags at
  // the standard exit of a Mex function. But at least Matlab 5.3 and
  // Matlab 6.5 leave the flags untouched on exits through mexErrMsgTxt.
  // TESTED FOR BCC, LCC, OWC, MSVC 2008 ONLY!
  
#if defined(__LCC__)           // Needs leading underscore
#define MCW_PC _MCW_PC
#define PC_64  _PC_64
#endif

  static unsigned int PC_Orig = 0;

  if (Cmd == 1) {
     PC_Orig = _control87(MCW_PC, 0);
     _control87(PC_64,   MCW_PC);
  } else {
     _control87(PC_Orig, MCW_PC);
  }
  
  return;
}
