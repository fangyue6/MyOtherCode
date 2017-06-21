function uTest_DNorm2(doSpeed)
% Automatic test: DNorm2
% This is a routine for automatic testing. It is not needed for processing and
% can be deleted or moved to a folder, where it does not bother.
%
% uTest_DNorm2(doSpeed)
% INPUT:
%   doSpeed: Optional logical flag to trigger time consuming speed tests.
%            Default: TRUE. If no speed test is defined, this is ignored.
% OUTPUT:
%   On failure the test stops with an error.
%
% Tested: Matlab 6.5, 7.7, 7.8, WinXP
% Author: Jan Simon, Heidelberg, (C) 2009-2010 matlab.THISYEAR(a)nMINUSsimon.de

% $JRev: R2.00j V:033 Sum:UpaTlH0z64kk Date:16-Oct-2010 00:14:37 $
% $License: BSD $
% $File: Tools\UnitTests_\uTest_DNorm2.m $
% History:
% 027: 25-Oct-2009 16:32, BUGFIX: Check of rejected bad inputs failed.

% Initialize: ==================================================================
if nargin == 0
   doSpeed = true;
end
minT = eps;

disp(['==== Test DNorm2:  ', datestr(now, 0)]);
disp(['Version: ', which('DNorm2')]);
fprintf('\n');

X  = [];
Y1 = DNorm2(X);
Y2 = DNorm2(X, 1);
Y3 = DNorm2(X, 2);
if isempty(Y1) && isempty(Y2) && isempty(Y3)
   disp('  ok: empty input');
else
   error('*** DNorm2: Failed on empty matrix');
end

dim0caught = 0;
try
   Y = DNorm2(X, 0);  %#ok<*NASGU>
catch
   dim0caught = dim0caught + 1;
end
try
   Y = DNorm2(1, 0);  %#ok<*NASGU>
catch
   dim0caught = dim0caught + 1;
end
try
   Y = DNorm2(rand(2), 0);  %#ok<*NASGU>
catch
   dim0caught = dim0caught + 1;
end
if dim0caught == 3
   disp('  ok: DNorm2(X, 0) rejected');
   lasterr('');
else
   error('*** DNorm2: Too friendly on Dim=0');
end

X = rand(2, 2);
Y = DNorm2(X);
Z = sqrt(sum(X .* X, 1));
if ~isEqualTol_L(Y, Z)
   error('*** DNorm2: Failed on 2x2 matrix, no Dim');
else
   disp('  ok: rand(2)');
end

Y = DNorm2(X, 2);
Z = sqrt(sum(X .* X, 2));
if ~isEqualTol_L(Y, Z)
   error('*** DNorm2: Failed on 2x2 matrix, Dim=2');
else
   disp('  ok: rand(2), DIM = 2');
end

X = rand(4, 5, 6, 7) - 0.5;
for iN = 1:4
   Y = DNorm2(X, iN);
   Z = sqrt(sum(X .* X, iN));
   if ~isEqualTol_L(Y, Z)
      error(['*** DNorm2: Failed on 4x5x6x7 array, Dim=', ...
            sprintf('%d', iN)]);
   end
   
   C     = {':', ':', ':', ':'};
   C{iN} = 1;
   X2    = X(C{:});
   Y     = DNorm2(X2, iN);
   if ~isequal(Y, abs(X2))
      C{iN} = '1';
      error(sprintf('*** DNorm2: Failed on [%s, %s, %s, %s] array, Dim=%d', ...
            C{:}, iN));  %#ok<SPERR>
   end
end
disp('  ok: All dims of: rand(4,5,6,7)');

% Test special treatment of size(X,N)==3:
for i = 1:3
   s    = [10, 10, 10];
   s(i) = 3;
   X    = rand(s);
   for j = 1:3
      Y = DNorm2(X, j);
      Z = sqrt(sum(X .* X, j));
      if ~isEqualTol_L(Y, Z)
         error(sprintf('*** DNorm2: Failed on %dx%dx%d matrix, Dim=%d', ...
            size(X), j));  %#ok<SPERR>
      end
   end
end
disp('  ok: 3D array with N=3 in any dimension');

Y = DNorm2([-Inf, Inf]);
if isinf(Y)
   disp('  ok: DNorm2([-Inf, Inf])');
else
   error('*** DNorm2: Failed on [-Inf, Inf] vector');
end

Y = DNorm2([1:10, NaN]);
if isnan(Y)
   disp('  ok: DNorm2([1:10, NaN])');
else
   error('*** DNorm2: Failed on [1:10, NaN] vector');
end

disp([char(10), '== Passed all tests: DNorm2(X, N) == sqrt(sum(X .* X, N))']);

% ==============================================================================
disp([char(10), 'Speed tests...']);

% Find a suiting number of loops:
if doSpeed
   % Number of loops adjusted to processor speed:
   X = rand(100, 100);
   
   iLoop     = 0;
   startTime = cputime;
   while cputime - startTime < 1.0
      X(1)  = rand;    % Impede the JIT acceleration
      a     = sqrt(sum(X .* X, 1));
      iLoop = iLoop + 1;
   end
   nLoops = 100 * ceil(iLoop / ((cputime - startTime) * 100));
   disp([sprintf('  %d', nLoops), ' loops on this machine, times in [sec].']);
else
   disp('  Use at least 2 loops (Displayed times are random!)');
   nLoops = 2;
end
drawnow;

% Change the value of the first element to impede the JIT acceleration:
pool = rand(1, nLoops);

disp('Vector 10000 x 1:');
X = rand(10000, 1);
tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = norm(X);
end
mnTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X));
end
m1Time = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(X.' * X);
end
m2Time = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 1);
end
cTime = toc;

disp(['  norm(X):              ', sprintf('%.2f', mnTime)]);
disp(['  sqrt(sum(X .* X, 1)): ', sprintf('%.2f', m1Time)]);
disp(['  sqrt(X.'' * X):        ', sprintf('%.2f', m2Time)]);
disp(['  DNorm2(X, 1):         ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mnTime), ' of NORM time']);
drawnow;

disp('Square matrix 100x100:');
X = rand(100, 100);
tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 1));
end
mTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 1);
end
cTime = toc;

% Compare with James Tursa's MTIMESX:
% X = reshape(X, 100, 1, 100);
% tic;
% for i = 1:nLoops
%    X(1) = pool(i);    % Impede the JIT acceleration
%    a    = sqrt(mtimesx(X, 't', X));
% end
% mtimesxTime = toc;

disp(['  sqrt(sum(X .* X, 1)): ', sprintf('%.2f', mTime)]);
% disp(['  sqrt(mtimesx(X, ''t'', X))): ', sprintf('%.2f', mtimesxTime)]);
disp(['  DNorm2(X):            ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 2));
end
mTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 2);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 2)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X, 2):         ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

disp('3D array 10x20x30:');
X = rand(10, 20, 30);
tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 1));
end
mTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 1);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 1)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X):            ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

disp('3D array 10x100x10:');
X = rand(10, 100, 10);
tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 1));
end
mTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 1);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 1)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X, 1):         ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

tic;
for i = 1:nLoops
   X(1) = rand;    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 2));
end
mTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 2);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 2)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X, 2):         ', sprintf('%.2f', cTime), '  ==> ', ...
   sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 3));
end
mTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 3);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 3)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X, 3):         ', sprintf('%.2f', cTime), '  ==> ', ...
   sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

disp('Matrix 10000x3:');
X = rand(10000, 3);
tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 1));
end
mTime = minT + toc;

tic;
for i = 1:nLoops
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 1);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 1)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X, 1):         ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

tic;
for i = 1:nLoops / 4
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 2));
end
mTime = minT + toc;

tic;
for i = 1:nLoops / 4
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 2);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 2)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X, 2):         ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);
drawnow;

disp('Matrix 3x10000:');
X = rand(3, 10000);
tic;
for i = 1:nLoops / 4
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = sqrt(sum(X .* X, 1));
end
mTime = minT + toc;

tic;
for i = 1:nLoops / 4
   X(1) = pool(i);    % Impede the JIT acceleration
   a    = DNorm2(X, 1);
end
cTime = toc;

disp(['  sqrt(sum(X .* X, 1)): ', sprintf('%.2f', mTime)]);
disp(['  DNorm2(X, 1):         ', sprintf('%.2f', cTime), '  ==> ', ...
      sprintf('%.1f%%', 100 * cTime / mTime), ' of Matlab time']);

% Bye:
fprintf('\nDNorm2 seems to work fine.\n');

return;

% ******************************************************************************
function Equal = isEqualTol_L(x, y, Tol)
% Compare two double arrays with absolute tolerance
% Equal = isEqualTol_L(x, y, [Tol])
% If the maximal difference between two double arrays is smaller than Tol, they
% are accepted as equal. If Tol is not specified, 10*eps is set as limit. For
% multiplied values SQRT(EPS) is a good choice. Use IsEqualRel for relative
% tolerance.
%
% As in Matlab's ISEQUAL, NaNs are treated as inequal, so
% isEqualTol_L(NaN, NaN) is FALSE. If you need comparable NaNs use
% ISEQUALWITHEQUALNANS, although this name is horrible and it is not
% existing in Matlab5.3.
%
% Use the MEX-script for drastic speedup, about 7 times. Currently the
% MEX file works on doubles only.
%
% Tested: Matlab 6.5, 7.7, 7.8, WinXP
% Author: Jan Simon, Heidelberg, (C) 2007-2010 matlab.THISYEAR(a)nMINUSsimon.de

% Was JRev: R0d V:023 Sum:FqP1dJr+CvYh Date:31-May-2010 00:25:38 $

% Initialize: ==================================================================
% Do the work: =================================================================
% Try the exact and fast comparison at first:
if isequal(x, y)
   Equal = true;
   return;
end

if nargin == 3
   Tol = abs(Tol);
else
   Tol = 10 * eps;
end

Equal = false;
if isequal(size(x), size(y))
   xMy = double(x) - double(y);   % No operations on SINGLEs in Matlab 6!
   
   % Same as "if all(abs(xMy(:)) <= Tol)", but faster:
   if all(or((abs(xMy) <= Tol), (x == y)))   % is FALSE for NaNs
       Equal = true;
   end
end

return;
