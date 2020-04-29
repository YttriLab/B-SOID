function varargout = RunLength(varargin)
% RUNLENGTH - Run-length en- and decoding
% Run-length encoding splits a vector into one vector, which contains the
% elements without neighboring repetitions, and a second vector, which
% contains the number of repetitions.
% This can reduce the memory for storing the data or allow to analyse sequences.
%
% Encoding: [B, N, BI] = RunLength(X)
% Decoding: X          = RunLength(B, N)
% INPUT / OUTPUT:
%   X:  Full input signal, row or column vector.
%       Types: (U)INT8/16/32/64, SINGLE, DOUBLE, LOGICAL, CHAR.
%   B:  Compressed data, neighboring elements with the same value are removed.
%       B and X have the same types.
%   N:  Number of repetitions of the elements of B in X as DOUBLE or UINT8 row
%       vector.
%   BI: Indices of elements in B in X as DOUBLE row vector.
%
% RunLength(X, 'byte') replies N as UINT8 vector. Sequences with more than 255
% elements are split into parts. This needs less memory and processing time.
%
% NOTES:
%   The comparison treats NaNs as equal.
%   You can find a lot of RLE tools in the FileExchange already. This C-Mex is
%   about 5 times faster than good vectorized M-versions.
%   The M-file RunLength_M contains vectorized and loop M-code for education.
%
% EXAMPLES:
% Encode and decode:
%   [b, n] = RunLength([8, 9, 9, 10, 10, 10, 11])
%   x      = RunLength(b, n)
%   % b = [8,9,10,11],  n = [1,2,3,1],  x = [8,9,9,10,10,10,11]
% Limit counter to 255:
%   [b, n] = RunLength(ones(1, 257), 'byte')
%   % b = [1, 1],  n = uint8([255, 2])
% LOGICAL input:
%   [b, n] = RunLength([true(257, 1); false])
%   % b = [true; false],  n = [257, 1]
% Find the longest sequence:
%   x          = floor(rand(1, 1e6) * 2);
%   [b, n, bi] = RunLength(x);
%   [longestRun, index] = max(n);
%   longestPos = bi(index);
%
% COMPILATION:
% Run this function once to compile the MEX file. Alternative method are
% explained in RunLength.c .
% Run the unit-test uTest_RunLength to test validity and speed!
%
% Tested: Matlab/64 7.8, 7.13, 8.6, 9.1, Win7/64
%         Compiler: LCC3.8, BCC5.5, OWC1.8, MSVC2008/2010
% Does not compile under LCC2.4 shipped with Matlab6.5!
% Assumed Compatibility: higher Matlab versions, Linux, MacOS.
%
% Author: Jan Simon, Heidelberg, (C) 2013-2017 matlab.2010(a)n(MINUS)simon.de

% $JRev: R-l V:011 Sum:tYFzC0o6wRYk Date:08-Apr-2017 18:41:21 $
% $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
% $UnitTest: uTest_RunLength $
% $File: Tools\GLMath\RunLength.m $
% History:
% 001: 23-Mar-2013 00:45, First version.
% 002: 29-Mar-2013 23:36, 3rd output BI for encoding.
% 011: 08-Apr-2017 18:39, 2nd output is a column vector if 1st output is one.
%      Thanks to The Cyclist for this suggestion.

% Initialize: ==================================================================
% Do the work: =================================================================
% This is a dummy code only, which compiles the C-function automatically:
Ok = InstallMex('RunLength.c', 'uTest_RunLength');
if ~Ok
   error('JSimon:RunLength:BadCompilation', 'Installation failed.');
end

[varargout{1:nargout}] = RunLength(varargin{:});

% return;
