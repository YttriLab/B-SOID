function [out1, out2, out3] = RunLength_M(in1, in2)
% RUNLENGTH_M - RLE coding as M-function for education
% Run-length encoding splits a vector into one vector, which contains the
% elements without neighboring repetitions, and a seconds vector, which
% contains the number of repetitions.
% This can reduce the memory for storing the data or allow to analyse sequences.
%
% This M-version is thought for educational purposes.
% The vectorized methods are fairly efficient and they can be inserted in a
% productive code, when the MEX cannot be used for any reasons.
% The MEX function is 4 to 5 times faster and can create N as UINT8.
%
% Encoding: [B, N, IB] = RunLength(X)
% Decoding: X          = RunLength(B, N)
% INPUT / OUTPUT:
%   X:  Full input signal, row or column vector.
%       Types: (U)INT8/16/32/64, SINGLE, DOUBLE, LOGICAL, CHAR.
%   B:  Compressed data, neighboring elements with the same value are removed.
%       B and X have the same types.
%   N:  Number of repetitions of the elements of B in X as DOUBLE or UINT8 row
%       vector.
%   IB: Indices of elements in B in X as DOUBLE row vector.
%
% NOTE: For more information and examples see RunLength.m.
%
% Tested: Matlab/64 7.8, 7.13, 8.6, 9.1, Win7/64
% Author: Jan Simon, Heidelberg, (C) 2013-2017 matlab.2010(a)n(MINUS)simon.de

% $JRev: R-f V:005 Sum:RVxUGzdUjK2X Date:08-Apr-2017 18:41:21 $
% $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
% $File: Tools\GLMath\RunLength_M.m $
% History:
% 001: 15-Mar-2013 09:30, First version.
% 005: 08-Apr-2017 18:32, 2nd output is a column vector if 1st output is one.

% Initialize: ==================================================================
% Global Interface: ------------------------------------------------------------
% Initial values: --------------------------------------------------------------
% Set to TRUE or FALSE manually to compare the speed:
vectorized = true;
%#ok<*UNRCH>  % Suppress MLint warnings about unreachable code

% Program Interface: -----------------------------------------------------------
% Check inputs:
switch nargin
   case 1
      doEncode = true;
   case 2
      doEncode = false;
      if ischar(in2)
         error('JSimon:RunLength_M:BadTypeIn2', ...
               '*** RunLength[m]: ''byte'' method not supported in M-version.');
      end
      if ~all(in2 == floor(in2)) || any(in2 < 0)
         error('JSimon:RunLength_M:BadValueN', ...
               '*** RunLength[m]: N must have non-negative integer values.');
      end
   otherwise
      error('JSimon:RunLength_M:BadNInput', ...
            '*** RunLength[m]: 1 or 2 inputs allowed');
end

% No cells, structs or objects:
if ~(isnumeric(in1) || islogical(in1) || ischar(in1))
   error('JSimon:RunLength_M:BadTypeInput1', ...
         '*** RunLength[m]: 1st input must be numeric, logical or char.');
end

% Fast return for empty inputs:
if isempty(in1)
   out1 = in1([]);
   if nargout == 2
      out2 = [];
   end
   return;
end

% Input must be a vector:
[s1, s2] = size(in1);
if ~ismatrix(in1) || (s1 ~= 1 && s2 ~= 1)
   error('JSimon:RunLength_M:BadShapeInput1', ...
         '*** RunLength[m]: 1st input must be a row or column vector.');
end

% User Interface: --------------------------------------------------------------
% Do the work: =================================================================
if doEncode                       % Encoding: [x] -> [b, n] --------------------
   x = in1(:);
   
   if vectorized                  % Vectorized: --------------------------------
      d = [true; diff(x) ~= 0];   % TRUE if values change
      b = x(d);                   % Elements without repetitions
      k = find([d', true]);       % Indices of changes
      n = diff(k);                % Number of repetitions
      
      if nargout == 3             % Reply indices of changes
         out3 = k(1:length(k) - 1);
      end
      
   else                           % Loop: --------------------------------------
      len = length(x);            % Output must have <= len elements
      b(len, 1) = x(1);           % Pre-allocate, dummy value to copy the class
      n   = zeros(1, len);        % Pre-allocate
      ib  = 1;                    % Cursor for output b
      xi  = x(1);                 % Remember first value
      ix  = 1;                    % Cursor for input x
      for k = 2:len               % Compare from 2nd to last element
         if x(k) ~= xi            % If value has changed
            b(ib) = xi;           % Store value in output
            n(ib) = k - ix;       % Store number of repetitions in output
            ib    = ib + 1;       % Increase the output cursor
            ix    = k;            % Initial index of the next run
            xi    = x(k);         % Value of the next run
         end
      end
      b(ib) = xi;                 % Flush last element
      n(ib) = len - ix + 1;
      
      b(ib + 1:len) = [];         % Crop unused elements in the output
      n(ib + 1:len) = [];
      
      if nargout == 3             % Reply indices of changes
         out3 = cumsum([1, n(1:len - s1)]);
      end
   end
   
   if s2 > 1                      % Output gets same orientation as input
      b = b.';
      n = n.';
   end
   out1 = b;
   out2 = n;
   
else                              % Decoding: [b, n] -> [x] ====================
   b = in1(:);                    % More convenient names for inputs
   n = in2;
   
   if vectorized                  % Vectorized: --------------------------------
      len   = length(n);          % Number of bins
      d     = cumsum(n);          % Cummulated run lengths
      index = zeros(1, d(len));   % Pre-allocate
      index(d(1:len-1)+1) = 1;    % Get the indices where the value changes
      index(1)            = 1;    % First element is treated as "changed" also
      index = cumsum(index);      % Cummulated indices
      x     = b(index);
      
   else                           % Loop: --------------------------------------
      len      = sum(n);          % Length of the output
      x(len,1) = b(1);            % Pre-allocate, dummy value to copy the class
      i1       = 1;               % Start at first element
      for k = 1:length(n)         % Loop over all elements of the input
         i2         = i1 + n(k);  % Start of next run
         x(i1:i2-1) = b(k);       % Repeated values
         i1         = i2;         % Set current start to start of next run
      end
   end
   
   if s2 > 1                      % Output gets same orientation as input
      x = x.';
   end
   out1 = x;
end
