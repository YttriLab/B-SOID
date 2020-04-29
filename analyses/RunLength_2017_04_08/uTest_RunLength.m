function uTest_RunLength(doSpeed)
% Automatic test: RunLength
% This is a routine for automatic testing. It is not needed for processing and
% can be deleted or moved to a folder, where it does not bother.
%
% uTest_RunLength(doSpeed)
% INPUT:
%   doSpeed: Optional logical flag to trigger time consuming speed tests.
%            Default: TRUE. If no speed test is defined, this is ignored.
% OUTPUT:
%   On failure the test stops with an error.
%   To check the circumstances, type "dbstop if error" and run the test again.
%   Then Matlab stops, when the error occurs.
%   Please send the error message and a description of your setup to the author.
%   Thanks!
%
% Tested: Matlab 6.5, 7.7, 7.8, 7.13, WinXP/32, Win7/64
% Author: Jan Simon, Heidelberg, (C) 2010-2016 matlab.2010(a)n(MINUS)simon.de

% $JRev: R-L V:037 Sum:1Mmp5VzPTWp8 Date:08-Apr-2017 18:41:21 $
% $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
% $File: Tools\UnitTests_\uTest_RunLength.m $
% History:
% 031: 29-Mar-2013 23:36, 3rd output BI for encoding.
% 035: 01-Feb-2016 00:07, Comparison with REPELEM.
% 037: 08-Apr-2017 18:32, Test column orientation of 2nd output.

% Initialize: ==================================================================
if nargin == 0
   doSpeed = true;
end
ErrID = ['JSimon:', mfilename];

% Existence of REPELEM:
matlabV    = [100, 1] * sscanf(version, '%d.', 2);
hasRepElem = (matlabV >= 806);

% Hello:
disp(['==== Test RunLength:  ', datestr(now, 0)]);
disp(['  Matlab:  ', version]);

whichRunLength = which('RunLength');
if isempty(whichRunLength)
   error(ErrID, 'RunLength function not found.');
end

[dummy, fcnName, fcnExt] = fileparts(whichRunLength);
fcnFile = [fcnName, fcnExt];

disp(['  Version: ', whichRunLength, char(10)]);

% Known answer tests: ----------------------------------------------------------
disp('== Known answer tests:');
typeList = {'char', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', ...
   'uint64', 'int64', 'double', 'single', 'logical'};
for iType = 1:length(typeList)
   aType = typeList{iType};
   
   aEmpty = myCast([], aType);
   [b, n] = RunLength(aEmpty);
   if ~isempty(b) || ~isempty(n) || ~isa(b, aType) || ~isa(n, 'double')
      error(ErrID, 'Failed for: ([]), class: %s', aType);
   end
   
   x = RunLength(aEmpty, []);
   if ~isempty(x) || ~isa(x, aType)
      error(ErrID, 'Failed for: ([], []), class: %s', aType);
   end
   
   x0     = myCast(1, aType);
   [b, n] = RunLength(x0);
   if ~isequal(b, x0) || ~isequal(n, 1) || ~isa(b, aType) || ~isa(n, 'double')
      error(ErrID, 'Failed for: ([1]), class: %s', aType);
   end
   
   x = RunLength(b, n);
   if ~isequal(x, x0) || ~isa(x, aType)
      error(ErrID, 'Failed for: ([1], [1]), class: %s', aType);
   end
  
   x0     = myCast(0:1, aType);
   [b, n] = RunLength(x0);
   if ~isequal(b, x0) || ~isequal(n, [1,1]) || ~isa(b, aType) ...
         || ~isa(n, 'double')
      error(ErrID, 'Failed for: ([0,1]), class: %s', aType);
   end
   
   x = RunLength(b, n);
   if ~isequal(x, x0) || ~isa(x, aType)
      error(ErrID, 'Failed for: ([0,1], [1,1]), class: %s', aType);
   end
   
   x0     = myCast([1,1], aType);
   [b, n] = RunLength(x0);
   if ~isequal(b, x0(1)) || ~isequal(n, 2) || ~isa(b, aType) ...
         || ~isa(n, 'double')
      error(ErrID, 'Failed for: ([1,1]), class: %s', aType);
   end
   
   x = RunLength(b, n);
   if ~isequal(x, x0) || ~isa(x, aType)
      error(ErrID, 'Failed for: ([1], [2]), class: %s', aType);
   end
   
   if not(strcmp(aType, 'logical'))  % LOGICAL has 0 and 1 only!
      x0 = myCast(0:127, aType);
      [b, n] = RunLength(x0);
      if ~isequal(b, x0) || ~isequal(n, ones(1,128)) || ~isa(b, aType) ...
            || ~isa(n, 'double')
         error(ErrID, 'Failed for: ([0:127]), class: %s', aType);
      end
      
      x = RunLength(b, n);
      if ~isequal(x, x0) || ~isa(x, aType)
         error(ErrID, 'Failed for: ([0:127], [ONES]), class: %s', aType);
      end
      
      x0 = myCast([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 2, 2], aType);
      [b, n] = RunLength(x0);
      if ~isequal(b, myCast([1, 2, 3, 4, 2], aType)) || ...
            ~isequal(n, [1, 2, 3, 4, 2]) || ~isa(b, aType) ...
            || ~isa(n, 'double')
         error(ErrID, 'Failed for: ([1,2,2,3,3,3,4,4,4,4,2,2]), class: %s', ...
            aType);
      end
      
      x = RunLength(b, n);
      if ~isequal(x, x0) || ~isa(x, aType)
         error(ErrID, 'Failed for: ([1,2,3,4,2], [1,2,3,4,2]), class: %s', ...
            aType);
      end
      
      % Test 0 runlength:
      b0 = myCast(1:10, aType);
      n0 = [1,2,3,0,3,2,1,0,4,5];
      x  = RunLength(b0, n0);
      [b, n] = RunLength(x);
      if ~isequal(b, b0(n0 ~= 0)) || ~isa(b, aType)
         error(ErrID, 'Failed for zeros in runlength, class: %s', aType);
      end
   end
   
   % Test coding with limitting counter to 255:
   for k = [1,2,3, 250:260, 500:520, 750:800]  % Around multiples of 255
      x0     = myCast(ones(1, k), aType);
      [b, n] = RunLength(x0, 'byte');
      x      = RunLength(b, n);
      if ~isequal(x, x0) || ~isa(x, aType) || ~isa(n, 'uint8')
         error(ErrID, 'Failed for: ones(1, %d), byte-limit, class: %s', ...
               k, aType);
      end
   end
   
   % Test column orientation:
   x0     = myCast([0;1;1], aType);
   [b, n] = RunLength(x0);
   if ~isequal(b, x0(1:2)) || ~isequal(n, [1;2]) || ~isa(b, aType) ...
         || ~isa(n, 'double')
      error(ErrID, 'Failed for: ([0;1;1]), class: %s', aType);
   end
   
   x = RunLength(b, n);
   if ~isequal(x, x0) || ~isa(x, aType)
      error(ErrID, 'Failed for: ([0;1], [1,2]), class: %s', aType);
   end
   
   x0     = myCast(randi([0,1], 1000, 1), aType);
   [b, n] = RunLength(x0);
   if ~isColumnVector(b) || ~isColumnVector(n)  || ~isa(b, aType) ...
         || ~isa(n, 'double')
      error(ErrID, 'Failed for: (large column), class: %s', aType);
   end
   
   x = RunLength(b, n);
   if ~isequal(x, x0) || ~isa(x, aType)
      error(ErrID, 'Failed for: ([0;1], [1,2]), class: %s', aType);
   end
   
   % Test 3rd output index:
   [b, n, bi] = RunLength([1, 2, 2, 3, 3, 3, 4]);
   if ~isequal(bi, [1, 2, 4, 7])
      error(ErrID, 'Bad index for [1, 2, 2, 3, 3, 3, 4], class: %s', aType);
   end
   
   [b, n, bi] = RunLength([1, 2, 2, 3, 3, 3, 4], 'byte');
   if ~isequal(bi, [1, 2, 4, 7])
      error(ErrID, ...
         'Bad index for [1, 2, 2, 3, 3, 3, 4] (byte), class: %s', aType);
   end
   
   [b, n, bi] = RunLength([2, 2, 3, 3, 3, 4, 5, 5]);
   if ~isequal(bi, [1, 3, 6, 7])
      error(ErrID, 'Bad index for [2, 2, 3, 3, 3, 4, 5, 5], class: %s', aType);
   end
   
   [b, n, bi] = RunLength([2, 2, 3, 3, 3, 4, 5, 5], 'byte');
   if ~isequal(bi, [1, 3, 6, 7])
      error(ErrID, ...
         'Bad index for [2, 2, 3, 3, 3, 4, 5, 5] (byte), class: %s', aType);
   end
   
   % Test Inf and NaN:
   if strcmp(aType, 'double') || strcmp(aType, 'single')
      x0 = myCast(NaN, aType);
      [b, n] = RunLength(x0);
      if ~isnan(b) || length(b) ~= 1 || ~isa(b, aType) || ~isa(n, 'double')
         error(ErrID, 'Failed for: (NaN), class: %s', aType);
      end
      
      x = RunLength(b, n);
      if ~isequal(isnan(x), isnan(x0)) || ~isa(x, aType)
         error(ErrID, 'Failed for: (NaN, [1]), class: %s', aType);
      end
      
      x0 = myCast([NaN, NaN], aType);
      [b, n] = RunLength(x0);
      if ~isnan(b) || length(b) ~= 1 || ~isa(b, aType) || ~isa(n, 'double') ...
            || ~isequal(n, 2)
         error(ErrID, 'Failed for: ([NaN,NaN]), class: %s', aType);
      end
      
      x = RunLength(b, n);
      if ~isequal(isnan(x), isnan(x0)) || ~isa(x, aType)
         error(ErrID, 'Failed for: (NaN, [2]), class: %s', aType);
      end
      
      x0 = myCast([NaN, 1, NaN, NaN, 2, 2], aType);
      [b, n] = RunLength(x0);
      if ~isequal(isnan(b), [true, false, true, false]) ...
            || ~isa(b, aType) || ~isa(n, 'double') ...
            || ~isequal(n, [1,1,2,2])
         error(ErrID, 'Failed for: ([NaN,1,NaN,NaN,2,2]), class: %s', aType);
      end
      
      x = RunLength(b, n);
      if ~isequal(isnan(x), isnan(x0)) || ~isa(x, aType)
         error(ErrID, ...
            'Failed for: ([NaN,1,NaN,NaN,2,2], [1,1,2,2]), class: %s', aType);
      end
      
      x0 = myCast([Inf, -Inf], aType);
      [b, n] = RunLength(x0);
      if ~isequal(b, x0) || ~isa(b, aType) || ~isa(n, 'double')
         error(ErrID, 'Failed for: ([Inf, -Inf]), class: %s', aType);
      end
      
      x = RunLength(b, n);
      if ~isequal(x, x0) || ~isa(x, aType)
         error(ErrID, 'Failed for: ([Inf, -Inf], [1,1]), class: %s', aType);
      end
      
      x0 = myCast([Inf, Inf, -Inf, -Inf, -Inf], aType);
      [b, n] = RunLength(x0);
      if ~isequal(b, [Inf, -Inf]) || ~isa(b, aType) || ~isa(n, 'double') ...
            || ~isequal(n, [2, 3])
         error(ErrID, 'Failed for: ([Inf,Inf,-Inf,-Inf,-Inf]), class: %s', ...
            aType);
      end
      
      x = RunLength(b, n);
      if ~isequal(x, x0) || ~isa(x, aType)
         error(ErrID, 'Failed for: ([Inf, -Inf], [2,3], class: %s', aType);
      end
   end
end
disp('  ok');

disp('== Random data tests:');  % ----------------------------------------------
iRndTest = 0;
for iType = 1:length(typeList)
   aType = typeList{iType};
      
   finTime = now + 0.5 / 86400;  % 2 seconds
   while now < finTime
      iRndTest = iRndTest + 1;
      n  = floor(rand * 200);
      x0 = myCast(floor(rand(1, n) * 32), aType);
      if rand > 0.5
         [b, n] = RunLength(x0);
      else
         [b, n] = RunLength(x0, 'byte');
      end
      x = RunLength(b, n);
      
      % If x0 is a 1-by-0 array, an 0-by-0 array is replied. In all other cases,
      % x must equal x0:
      if ~isequal(x, x0) && ~(isempty(x) && isempty(x0))
         fprintf('Bad reply:\n  class: %s\n', aType);
         fprintf('\n  b: ');
         fprintf('%d ', x0);
         fprintf('\n  b: ');
         fprintf('%d ', b);
         fprintf('\n  n: ');
         fprintf('%d ', n);
         
         error(ErrID, 'Failed for: random test data.');
      end
   end
end
fprintf('  Number of tests: %d\n', iRndTest);
disp('  ok');

disp('== Check catching or bad input:');  % ------------------------------------
tooLazy = false;
try
   [b, n]  = RunLength([], 1);  %#ok<*ASGLU>
   tooLazy = true;
catch %#ok<*CTCH>
end
if tooLazy
   error(ErrID, 'Bad input not detected: ([], 1)');
end

try
   [b, n]  = RunLength(1, []);
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, 'Bad input not detected: (1, [])');
end

try
   [b, n]  = RunLength(1:2, 1);
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, 'Bad input not detected: ([1,2], 1)');
end

try
   [b, n]  = RunLength({1}, 1);
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, 'Bad input not detected: ({1}, 1)');
end

try
   [b, n]  = RunLength([], 1);
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, 'Bad input not detected: ([], 1)');
end

try
   [b, n, q1, q2] = RunLength(1:4);
   tooLazy        = true;
catch
end
if tooLazy
   error(ErrID, '4 outputs not rejected for encoding.');
end

try
   [b, n]  = RunLength(1:2, 1:2);
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, '2 outputs not rejected for decoding.');
end

try
   x       = RunLength(1:2, int8(1:2));  %#ok<*NASGU> % Must be double or uint8
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, '2nd input with bad type accepted?!');
end

try
   x = RunLength(1:2, [2, -1]);
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, 'Negative runlength accepted?!');
end

try
   [b, n]  = RunLength(rand(2, 3));
   tooLazy = true;
catch
end
if tooLazy
   error(ErrID, 'Matrix input accepted?!');
end

disp('  ok');

% Speed: -----------------------------------------------------------------------
if doSpeed
   % The M-code does not contain checks of types, sizes and values of the
   % inputs, but this matters for small data mainly.
   disp('== Speed:');
   
   % Average run lengths:
   avgWidth = 15;
   
   for Len = [1e3, 1e6]
      found = false;
      while ~found
         nn = ceil(rand(1, 1000 + ceil(Len / avgWidth)) * (avgWidth * 2 - 1));
         ns = cumsum(nn);
         kk = find(ns >= Len);  % No FIND(x,1) in Matlab 6.5
         if any(kk)
            found   = true;
            x       = zeros(1, Len);
            k       = kk(1) - 1;
            nk      = ns(k);
            x(1:nk) = RunLength(rem(1:k, 2), nn(1:k));
         end
      end
      
      % Number of loops:
      nLoop = 10e7 / Len;

      % Factor to get MB/sec:
      TimeToMB = nLoop * Len / 1e6;

      [b, n] = RunLength(x);
      fprintf('Data: [1 x %g] double, mean runlength: %.3f\n', Len, mean(n));
      
      fprintf('  C-Mex:\n');
      tic;
      for k = 1:nLoop
         [b, n] = RunLength(x);
      end
      mex_time = toc;
      fprintf('    Encoding: %7.1f MB/sec\n', TimeToMB / mex_time);
      
      tic;
      for k = 1:nLoop
         x2 = RunLength(b, n);
      end
      mex_time = toc;
      fprintf('    Decoding: %7.1f MB/sec\n', TimeToMB / mex_time);
      
      fprintf('  C-Mex with UINT8 counter:\n');
      tic;
      for k = 1:nLoop
         [b255, n255] = RunLength(x, 'byte');
      end
      mexByte_time = toc;
      fprintf('    Encoding: %7.1f MB/sec\n',  TimeToMB / mexByte_time);
      
      tic;
      for k = 1:nLoop
         x2 = RunLength(b255, n255);
      end
      mexByte_time = toc;
      fprintf('    Decoding: %7.1f MB/sec\n',  TimeToMB / mexByte_time);
      
      fprintf('  M-code vectorized:\n');
      tic;
      for k = 1:nLoop
         [b, n] = local_Encode_Vect(x);
      end
      mVect_time = toc;
      fprintf('    Encoding: %7.1f MB/sec\n', TimeToMB / mVect_time);
            
      tic;
      for k = 1:nLoop
         x2 = local_Decode_Vect(b, n);
      end
      mVect_time = toc;
      fprintf('    Decoding: %7.1f MB/sec\n', TimeToMB / mVect_time);
      
      fprintf('  M-code loop:\n');
      tic;
      for k = 1:nLoop * 0.5
         [b, n] = local_Encode_Loop(x);
      end
      mLoop_time = toc;
      fprintf('    Encoding: %7.1f MB/sec\n', 0.5 * TimeToMB / mLoop_time);

      tic;
      for k = 1:nLoop * 0.25  % Less loops, because this is slow
         x2 = local_Decode_Loop(b, n);
      end
      mLoop_time = toc;
      fprintf('    Decoding: %7.1f MB/sec\n', 0.25 * TimeToMB / mLoop_time);
      
      if hasRepElem
         fprintf('  REPELEM:\n');
         tic;
         for k = 1:nLoop
            x2 = repelem(b, n);
         end
         reVect_time = toc;
         fprintf('    Decoding: %7.1f MB/sec\n', TimeToMB / reVect_time);
      end
   end
end

% Goodbye: ---------------------------------------------------------------------
fprintf('\n%s passed the tests successfully.\n', fcnFile);

% return;

% ******************************************************************************
function x = myCast(x, ClassName)
%  Support the CAST function for the old Matlab 6.5:
if sscanf(version, '%f', 1) > 7.0
   x = cast(x, ClassName);
else
   if strcmp(ClassName, 'logical')
      x = feval(ClassName, x ~= 0);
   else
      x = feval(ClassName, x);
   end
end

% return;


% ******************************************************************************
function Tf = isColumnVector(X)
Tf = (length(X) <= 1) || (ndims(X) == 2 && size(X, 1) == numel(X));  %#ok<ISMAT>
% return;

% ******************************************************************************
function x = local_Decode_Vect(b, n)

% Vectorized:
len   = length(n);
d     = cumsum(n);
index = zeros(1, d(len));
index(d(1:len-1)+1) = 1;
index(1)            = 1;
index = cumsum(index);
x     = b(index);
% return;

% ******************************************************************************
function x = local_Decode_Loop(b, n)

% Loop:
len      = sum(n);
x(1:len) = b(1);   % Dummy value to copy the class
i1   = 1;
for k = 1:length(n)
   i2         = i1 + n(k);
   x(i1:i2-1) = b(k);
   i1         = i2;
end
% return;

% ******************************************************************************
function [b, n] = local_Encode_Vect(x)

% Vectorized:
d = [true, diff(x) ~= 0];
b = x(d);
n = diff(find([d, true]));
% return;

% ******************************************************************************
function [b, n] = local_Encode_Loop(x)

% Loop:
len = length(x);
b(1:len) = x(1);  % Pre-allocate, dummy value to copy the class
n   = zeros(1, len);
ib  = 1;
xi  = x(1);
ix  = 1;
for k = 2:len
   if x(k) ~= xi
      b(ib) = xi;
      n(ib) = k - ix;
      ib    = ib + 1;
      ix    = k;
      xi    = x(k);
   end
end
b(ib) = xi;
n(ib) = len - ix + 1;

b(ib + 1:len) = [];   % Crop unused elements
n(ib + 1:len) = [];
% return;
