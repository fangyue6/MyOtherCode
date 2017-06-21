function Y = DNorm2(X, n)
% Fast euclidian norm over N.th dimension of a DOUBLE array
% Y = DNorm2(X, N)
% INPUT:
%   X: Real DOUBLE array.
%   N: Dimension to operate on.
% OUTPUT:
%   Y: Euclidian norm over N.th dimension: Y = sqrt(sum(X .* X, N)).
%      Length of N.th dimension of Y is 1.
%      NaN's are considered.
%
% Matlab's built-in function NORM is fast for vectors, but for matrices the
% matrix norm is replied. Other efficient methods for vectors(!):
%   Y = sqrt(sum(X .* X));
%   Y = sqrt(X * X');       % row vectors, faster than DNorm2!
% And James Tursa's MTIMESX is very fast for vectors also and can operate on
% the 1st dimension of arrays also:
%   X = rand(100, 100);  X = reshape(X, 100, 1, 100);
%   Y = sqrt(mtimesx(X, 't', X));
% But for arrays DNorm2 is faster, and I do not see a way to apply MTIMESX for
% trailing dimensions without time-consuming transpositions.
%
% COMPILATION:
%   This function must be compiled before using:
%     mex -O DNorm2.c
%   See DNorm2.c for detailed instructions.
%
% TEST: Run uTest_DNorm2 to test validity and speed.
%
% NOTES: See DNorm2.c for strategies to optimize processing speed depending
%   on the size of X.
%
% Tested: Matlab 6.5, 7.7, 7.8, WinXP, 32bit
%         Compiler: LCC2.4/3.8, BCC5.5, OWC1.8, MSVC2008
% Assumed Compatibility: higher Matlab versions, Mac, Linux, 64bit
% Author: Jan Simon, Heidelberg, (C) 2006-2010 matlab.THISYEAR(a)nMINUSsimon.de

% $JRev: R2.00e V:016 Sum:gLGZgPacp9Gw Date:15-Oct-2010 12:12:48 $
% $License: NOT_RELEASED $
% $UnitTest: uTest_DNorm2 $
% $File: Tools\GLMath\DNorm2.m $
% History:
% 009: 09-Aug-2006 00:31, Renamed: SqrtSumQuad -> DNorm2.

% Initialize: ==================================================================
persistent Done
if isempty(Done)
   % Use this Matlab script, but the MEX script would be much faster.
   fprintf(['??? ', mfilename, ': Cannot find compiler MEX file.', ...
      ' Slow Matlab version is used!\n']);
   Done = 1;
end

% Do the work: =================================================================
% Prefer the MEX function! But this works at least:
if nargin == 1
  size1 = find(size(X) ~= 1);
  if isempty(size1)
     n = 1;
  else
     n = size1(1);
  end
end

Y = sqrt(sum(X .* X, n));

return;
