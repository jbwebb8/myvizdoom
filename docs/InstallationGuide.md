# Installation and Setup Guide

## ViZDoom installation and build instructions (Ubuntu 16.04) 
### Summary
If you plan to use Anaconda and have not yet installed it, do so now. Otherwise, ViZDoom will depend on system packages and crash if installed prior to Anaconda. 

Be sure that you have installed all prerequisites for both ViZDoom and the original ZDoom. 

The following is a summary using Anaconda3 on Ubuntu 16.04: 

```
# zdoom dependencies 
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \ 
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \ 
libopenal-dev timidity libwildmidi-dev  

# create new virtual env in anaconda 
$ conda create --name vizdoom boost numpy scipy scikit-image scikit-learn tqdm mkl pip 
$ source activate vizdoom 

# install theano, lasagne, and tensorflow (see below) 
$ pip install --upgrade --user https://github.com/Theano/Theano/archive/master.zip 
$ pip install --upgrade --user https://github.com/Lasagne/Lasagne/archive/master.zip 
$ pip install --ignore-installed --upgrade $TF_PYTHON_URL 

# clone vizdoom repo 
$ git clone https://github.com/mwydmuch/ViZDoom 
$ cd ViZDoom 

# build binaries 
$ cmake -DCMAKE_BUILD_TYPE=Release \ 
-DBUILD_PYTHON3=ON \ 
-DPYTHON_INCLUDE_DIR=/path_to_conda/anaconda3/include/python3.6m \ 
-DPYTHON_LIBRARY=/path_to_conda/anaconda3/lib/libpython3.6m.so \ 
-DPYTHON_EXECUTABLE=/path_to_conda/anaconda3/bin/python3 \ 
-DBOOST_PYTHON3_LIBRARY=/path_to_conda/anaconda3/lib/libboost_python3.so \ 
-DNUMPY_INCLUDES=/path_to_conda/anaconda3/lib/python3.6/site-packages/numpy/core/include 
$ make –j 4 # allow four jobs at once 

# copy python package to site-packages 
$ cp /path_to_vizdoom/ViZDoom/bin/python3/pip_package/. \ 
/path_to_conda/anaconda3/lib/python3.6/site-packages/vizdoom 
```
And on OS X 10.11: 

```
# create new virtual env 
conda create --name vizdoom boost numpy scipy scikit-image scikit-learn tqdm mkl pip 
source activate vizdoom 

# install neural network packages 
pip install --upgrade --user https://github.com/Theano/Theano/archive/master.zip 
pip install --upgrade --user https://github.com/Lasagne/Lasagne/archive/master.zip 
pip install --ignore-installed --upgrade $TF_PYTHON_URL # Py3.6, CPU only: https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl 

# install cmake and system boost and boost-python 
brew install cmake 
brew install boost 
brew install boost-python --with-python3 

# if getting error installing boost-python, reset directory to sticky 
sudo chmod +t /private/tmp/boost_interprocess/ 

# note "Boost", not "BOOST"; also note that directory depends on version 
# do NOT set flags for FMOD EX libraries; it will fail in Anaconda env 
cmake -DCMAKE_BUILD_TYPE=Release \ 
-DCMAKE_OSX_ARCHITECTURES=x86_64 \ 
-DBUILD_PYTHON3=ON \ 
-DPYTHON_INCLUDE_DIR=$HOME/miniconda3/envs/vizdoom/include/python3.6m \ 
-DPYTHON_LIBRARY=$HOME/miniconda3/envs/vizdoom/lib/libpython3.6m.dylib \ 
-DPYTHON_EXECUTABLE=$HOME/miniconda3/envs/vizdoom/bin/python3 \ 
-DBoost_PYTHON3_LIBRARY=/usr/local/Cellar/boost-python/1.64.0/lib/libboost_python3.dylib \ 
-DNUMPY_INCLUDES=$HOME/miniconda3/lib/python3.6/site-packages/numpy/core/include 
$ make –j 4 # allow four jobs at once 

# copy python package to site-packages 
cp -r bin/python3/pip_package/ $HOME/path_to_conda/envs/vizdoom/lib/python3.6/site-packages/vizdoom 
```

### Installation in system environment  (Ubuntu 16.04) 

Open Terminal and change to directory in which you would like ViZDoom to operate. 

Clone ViZDoom into the current directory: 

```
$ git clone https://github.com/mwydmuch/ViZDoom 
$ cd ViZDoom 
```
 
pip install for either python 2 or 3 (note: this may take a few minutes): 

```
# notice the "." 
$ sudo pip install .  #python 2.7 
$ sudo pip3 install . #python 3.x 
```

Compile binaries from root directory (i.e ViZDoom/). (Note that this is in contrast to ZDoom, where the make files are generated in the root directory but binaries built from a separate "build" directory.) Default is BUILD_PYTHON=OFF, BUILD_PYTHON3=OFF, and BUILD_JAVA=OFF. To build wrappers, set desired flags to true. The compiler has difficulty finding the Python executable, include directory, and library, so those must be set with the flags DPYTHON_EXECUTABLE:FILEPATH, DPYTHON_INCLUDE_DIR:PATH, and DPYTHON_LIBRARY:FILEPATH. Here is an example building in Java and Python3: 

```
$ cmake -DCMAKE_BUILD_TYPE=Release \ -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3.5 \ -DPYTHON_INCLUDE_DIR:PATH=/usr/include/python3.5 \ -DPYTHON_LIBRARY:FILEPATH=/usr/lib/python3.5 \ -DBUILD_PYTHON3=ON \ 
-DBUILD_JAVA=ON 
$ make 
```

### Installation in Anaconda environment (Ubuntu 16.04) 
```
# zdoom dependencies 
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \ 
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \ 
libopenal-dev timidity libwildmidi-dev  

# create new virtual env in anaconda 
$ conda create --name vizdoom boost numpy scipy scikit-image scikit-learn tqdm mkl pip 
$ source activate vizdoom 

# install theano, lasagne, and tensorflow (see below) 
$ pip install --upgrade --user https://github.com/Theano/Theano/archive/master.zip 
$ pip install --upgrade --user https://github.com/Lasagne/Lasagne/archive/master.zip 
$ pip install --ignore-installed --upgrade $TF_PYTHON_URL 

# clone vizdoom repo 
$ git clone https://github.com/mwydmuch/ViZDoom 
$ cd ViZDoom 

# build binaries 
$ cmake -DCMAKE_BUILD_TYPE=Release \ 
-DBUILD_PYTHON3=ON \ 
-DPYTHON_INCLUDE_DIR=/path_to_conda/anaconda3/include/python3.6m \ 
-DPYTHON_LIBRARY=/path_to_conda/anaconda3/lib/libpython3.6m.so \ 
-DPYTHON_EXECUTABLE=/path_to_conda/anaconda3/bin/python3 \ 
-DBOOST_PYTHON3_LIBRARY=/path_to_conda/anaconda3/lib/libboost_python3.so \ 
-DNUMPY_INCLUDES=/path_to_conda/anaconda3/lib/python3.6/site-packages/numpy/core/include 
$ make –j 4 # allow four jobs at once 

# copy python package to site-packages 
$ cp /path_to_vizdoom/ViZDoom/bin/python3/pip_package/. \ 
/path_to_conda/anaconda3/lib/python3.6/site-packages/vizdoom 
 ```

### Installation in Anaconda environment (OS X 10.11) 
A summary of commands: 

```
# create new virtual env 
$ conda create --name vizdoom boost numpy scipy scikit-image scikit-learn tqdm mkl pip 
$ source activate vizdoom 

# install neural network packages into new virtual env 
$ pip install --upgrade --user https://github.com/Theano/Theano/archive/master.zip 
$ pip install --upgrade --user https://github.com/Lasagne/Lasagne/archive/master.zip 
$ pip install --ignore-installed --upgrade $TF_PYTHON_URL # Py3.6, CPU only: https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl 

# install rest of ViZDoom out of virtual env 
$ source deactivate vizdoom 

# install cmake and system boost and boost-python 
$ brew install cmake boost boost-python --with-python3 sdl2 wget 

# if getting error installing boost-python, reset directory to sticky 
$ sudo chmod +t /private/tmp/boost_interprocess/ 

# clone vizdoom repo 
$ git clone https://github.com/mwydmuch/ViZDoom 
$ cd ViZDoom 

# note "Boost", not "BOOST"; also note that directory depends on version 
# do NOT set flags for FMOD EX libraries; it will fail in Anaconda env 
$ cmake -DCMAKE_BUILD_TYPE=Release \ 
-DCMAKE_OSX_ARCHITECTURES=x86_64 \ 
-DBUILD_PYTHON3=ON \ 
-DPYTHON_INCLUDE_DIR=$HOME/path_to_conda/envs/vizdoom/include/python3.6m \ 
-DPYTHON_LIBRARY=$HOME/path_to_conda/envs/vizdoom/lib/libpython3.6m.dylib \ 
-DPYTHON_EXECUTABLE=$HOME/path_to_conda/envs/vizdoom/bin/python3 \ 
-DBoost_PYTHON3_LIBRARY=/usr/local/Cellar/boost-python/1.64.0/lib/libboost_python3.dylib \ 
-DNUMPY_INCLUDES=$HOME/path_to_conda/lib/python3.6/site-packages/numpy/core/include 
$ make –j 4 

# fix broken alias (optional) 
$ rm bin/vizdoom 
$ ln -s vizdoom.app/Contents/MacOS/vizdoom bin/vizdoom 

# copy python package to site-packages 
$ cp -r bin/python3/pip_package/ $HOME/path_to_conda/envs/vizdoom/lib/python3.6/site-packages/vizdoom 
```

A few subtleties on OS X: 
- Install and link to Boost and Boost-Python libraries on system rather than in Anaconda environment. For some reason, the compiler has problems linking Boost in Anaconda. 
- System permission errors can lead to obscure errors. If you have difficulty with Boost output,  try changing the permissions of /private/tmp/boost_interprocess.  
- Do not change the flags above when using CMake. Adding the FMOD library filepaths, for example, lead to errors when installing in Anaconda. (The compiler never links it correctly, but seems to avoid installing the broken link when on system.) 

### Testing installation 
Run a basic python example to test for successful installation: 

```
$ cd examples/python 
$ python basic.py 
```
 
Now a screen should appear with an agent randomly shooting at a monster on the back wall. If so, then success! 

## Slade installation (Ubuntu 16.04) 
See [wiki](http://slade.mancubus.net/index.php?page=wiki&wikipage=Installation) for more details 

Download DRD Team Package Repository to package sources: 

```bash
$ sudo apt-add-repository 'deb http://debian.drdteam.org/ stable multiverse' 
$ wget -O- http://debian.drdteam.org/drdteam.gpg | sudo apt-key add - 
```

Install Slade: 

```bash
$ sudo apt update 
$ sudo apt install slade 
```

Initial setup: 
1. Use temp folder (default) 
2. Add .wad file to base resource archive. This should be located at `ViZDoom/bin/freedoom2.wad`. 
3. Ignore path for nodebuilder (ZDBSP) and click Finish. 

Download and compile ACS compiler: 

```bash
$ git clone https://github.com/rheit/acc.git 
$ cd acc 
$ make 
```

Point Slade to ACC executable. Edit > Preferences. Select Scripting > ACS. Add executable to path, which should be located in acc folder extracted from github. (Note: the path to the acc executable cannot have spaces (in directories or filenames); otherwise, compiling will fail with "error 2".) 

## Troubleshooting
**Error**: Cannot find Boost Python3: 

```bash
Installing collected packages: vizdoom 
  Running setup.py install for vizdoom ... error 
… 
CMake Error at CMakeLists.txt:237 (MESSAGE): 
      Could not find boost python3 
```
 
**Solution**: Install boost python package. If switching from system to conda env, then make sure that you have installed boost in the conda env: 

```bash
$ conda install –c anaconda boost=1.61.0 # python 3 package for conda 
$ pip install boost   # python 2.7 
$ pip3 install boost # python 3.x 
```

**Error**: Cannot find dependencies while running setup.py in Anaconda. 

```bash
CMakeFiles/Makefile2:181: recipe for target 'CMakeFiles/libvizdoom_python3.dir/all' failed 
    make[1]: *** [CMakeFiles/libvizdoom_python3.dir/all] Error 2 
    Makefile:83: recipe for target 'all' failed 
    make: *** [all] Error 2 
… 
subprocess.CalledProcessError: Command '['make', '-j', '7']' returned non-zero exit status 2. 

    ---------------------------------------- 
Command "/home/james/anaconda3/bin/python -u -c "import setuptools, tokenize;__file__='/tmp/pip-ak17nt5z-build/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" install --record /tmp/pip-fb5i566p-record/install-record.txt --single-version-externally-managed --compile" failed with error code 1 in /tmp/pip-ak17nt5z-build/ 
```

**Solution**: As of 4/25/17, pip install does not work with Anaconda. Instead, you must build ViZDoom manually from the start (see Issue #178 and Issue #181). If you have previously tried to build with pip, first clear the Cmake cache: 

```bash
$ rm CMakeCache.txt 
```

or just  

```bash
$ ./cmake_clean.sh 
```

Next, manually build similar to the following Cmake commands: 

```bash
$ cmake -DCMAKE_BUILD_TYPE=Release \ 
-DBUILD_PYTHON3=ON \ 
-DPYTHON_INCLUDE_DIR=/path_to_conda/anaconda3/include/python3.6m \ 
-DPYTHON_LIBRARY=/path_to_conda/anaconda3/lib/libpython3.6m.so \ 
-DPYTHON_EXECUTABLE=/path_to_conda/anaconda3/bin/python3 \ 
-DBOOST_PYTHON3_LIBRARY=/path_to_conda/anaconda3/lib/libboost_python3.so \ 
-DNUMPY_INCLUDES=/path_to_conda/anaconda3/lib/python3.6/site-packages/numpy/core/include 
```

where path_to_conda represents the path to the Anaconda directory (for me, /home/james). If using a virtual environment in Anaconda, then add envs/env_name/ after anaconda3/ to lines above. Then make the binary files and copy the pip package to the Anaconda library. 

```bash
$ make –j 4 # allow four jobs at once 
$ cp /path_to_vizdoom/ViZDoom/bin/python3/pip_package/. \ 
/path_to_conda/anaconda3/lib/python3.6/site-packages/vizdoom 
```

**Error**: C++ library not found. 

```bash
ImportError: /path_to_conda/anaconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found 
```
 
**Solution**: GCC needs to be installed within the Anaconda environment. 

```bash
$ conda install libgcc 
```
 
**Error**: Memory allocation fault in C after manually installing Python pip_package into Anaconda environment. 

Running any python program, such as python basic.py, that imports vizdoom leads to an error message: 

```bash
$ python 
… 
>>> from vizdoom import * 
Segmentation fault 
$ python 
… 
>>> from vizdoom import * 
terminate called after throwing an instance of 'std::bad_alloc' 
```

These somewhat cryptic messages imply that memory was improperly allocated in the underlying C language that comprises ViZDoom, but it's unclear exactly how. By using Python Debugger (python –m pdb \<program\>) and/or verbose mode (python –v), it is noted that \_\_init\_\_.py does point the interpreter to vizdoom.so in the package, but then something goes awry as soon as it is read. 

**Solution**: The only solution on the GitHub site (Issue #175) a) does not use Anaconda and b) suggests reinstalling the entire OS. We'd rather not resort to such measures if we can help it. The error is likely due to a bug in the compilation process with the manual install—perhaps a missing or incorrect flag, or using a dependency that is different in the Anaconda environment. Whatever the reason, utilizing the Ubuntu system environment, and the compatible pip install that comes with setup.py in the ViZDoom root directory, seems to fix the issue. Thus create and install the pip package for Python in the system environment, then copy it over to Anaconda. 

First, open .bashrc in the home folder (use Ctrl+H to show hidden files). Comment out the path to the Anaconda, which is probably at the bottom: 

```bash
#export PATH="/path_to_conda/anaconda3/bin:$PATH" 
```
 
Save .bashrc and open a new Terminal session. Verify that the default path is now to the system python: 

```bash
$ which python 
/usr/bin/python 
$ python 
Python 2.7.12 (default, Nov 19 2016, 06:48:10)  
[GCC 5.4.0 20160609] on linux2 
Type "help", "copyright", "credits" or "license" for more information. 
>>>  
```

**Solution A**: Now follow the directions for installing ViZDoom on the system with pip. Check that the packages were properly installed: 

```bash
$ ls /usr/local/lib/python2.7/dist-packages/vizdoom # python2.7 
bots.cfg       __init__.py   scenarios  vizdoom.pk3 
freedoom2.wad  vizdoom       vizdoom.so 
$ ls /usr/local/lib/python3.5/dist-packages/vizdoom # python3 
bots.cfg       __init__.py  scenarios   vizdoom.pk3 
freedoom2.wad  vizdoom      vizdoom.so 
```

Now copy the contents of package(s) to the site-packages directory in Anaconda; for example: 

```bash
$ cp /usr/local/lib/python3.5/dist-packages/vizdoom/. \ 
/path_to_conda/anaconda3/lib/python3.6/site-packages/vizdoom 
```

Now add the path to Anaconda back into .bashrc by deleting # (or adding the full export PATH=… if deleted) and saving the file. Open a new Terminal and verify that Anaconda is default again: 

```bash
$ which python 
/home/james/anaconda3/bin/python 
```

Try to import the vizdoom module in the Python shell. No errors should be thrown. 

```bash
$ python 
Python 3.6.0 |Anaconda custom (64-bit)| (default, Dec 23 2016, 12:22:00)  
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux 
Type "help", "copyright", "credits" or "license" for more information. 
>>> from vizdoom import * 
>>> 
```

Confirm that ViZDoom works properly by running basic.py. 

**Solution B**: If this does not work (may throw an ImportError due to version mismatch), then try the following. Deactivate the anaconda environment as before (commenting out export PATH in ./bashrc), but instead of installing ViZDoom with pip, manually build it and still point cmake to the Anaconda python: 

```bash
$ cmake -DCMAKE_BUILD_TYPE=Release \ 
-DBUILD_PYTHON3=ON \ 
-DPYTHON_INCLUDE_DIR=/path_to_conda/anaconda3/include/python3.6m \ 
-DPYTHON_LIBRARY=/path_to_conda/anaconda3/lib/libpython3.6m.so \ 
-DPYTHON_EXECUTABLE=/path_to_conda/anaconda3/bin/python3 \ 
-DBOOST_PYTHON3_LIBRARY=/path_to_conda/anaconda3/lib/libboost_python3.so \ 
-DNUMPY_INCLUDES=/path_to_conda/anaconda3/lib/python3.6/site-packages/numpy/core/include 
$ make -j 4 # or up to (cpu cores - 1) 
```

remembering to replace /path_to_conda/ with the path (e.g. /home/james/) and adding envs/env_name/ after anaconda3/ if using a virtual environment. Now copy the contents of pip_package to the site packages folder in Anaconda as above.  

Note: system boost was also upgraded during the process of figuring out Solution B. It probably did not have anything to do with it, but in case Solution B is not working, you can try upgrading the system boost to 1.62 by following these instructions. Boost dynamic libraries seem to be responsible for the underlying issues of installing ViZDoom in the Anaconda environment. 

**Error**: Broken link to mkl library. 

```bash
$ python learning_theano_train.py -h 
Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so. 
```

**Solution**: mkl comes pre-installed with Anaconda, but sometime the links to the libraries become convoluted as more packages are installed with conflicting dependencies.  

```bash
$ conda list 
# packages in environment at /home/james/anaconda3: 
# 
… 
mkl                       2017.0.1                      0   
mkl-service               1.1.2                    py36_3   
... 
```

Solve these conflicts by updating mkl in the current environment: 

```bash
$ conda update mkl 
```

or Anaconda in root. However, this may downgrade other important packages, like libgcc. 

```bash
$ conda update anaconda 
The following packages will be UPDATED: 

    anaconda:     custom-py36_0      --> 4.3.1-np111py36_0 
    … 

 The following packages will be DOWNGRADED due to dependency conflicts: 
    ... 

    libgcc:       5.2.0-0            --> 4.8.5-2 
    …          
```

Leading to conflicts with the vizdoom package: 

```bash
Traceback (most recent call last): 
  File "learning_theano_train.py", line 15, in <module> 
    from vizdoom import * 
  File "/home/james/anaconda3/lib/python3.6/site-packages/vizdoom/__init__.py", line 1, in <module> 
    from .vizdoom import __version__ as __version__ 
ImportError: /home/james/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /home/james/anaconda3/lib/python3.6/site-packages/vizdoom/vizdoom.so) 
```

You can check to see that the version GLIBCXX_3.4.20 is indeed missing: 

```bash
$ strings /path_to_conda/anaconda3/lib/libstdc++.so.6 | grep GLIBCXX 
GLIBCXX_3.4 
GLIBCXX_3.4.1 
GLIBCXX_3.4.2 
... 
GLIBCXX_3.4.19 
GLIBCXX_FORCE_NEW 
GLIBCXX_DEBUG_MESSAGE_LENGTH 
```

Re-install libgcc to update to the most recent version on Anaconda: 

```bash
$ conda install libgcc 
… 
$ $ strings /path_to_conda/anaconda3/lib/libstdc++.so.6 | grep GLIBCXX 
GLIBCXX_3.4 
GLIBCXX_3.4.1 
GLIBCXX_3.4.2 
... 
GLIBCXX_3.4.19 
GLIBCXX_3.4.20 
GLIBCXX_3.4.21 
```

Alternatively, 

**Error**: Issue importing Lasagne and/or Theano module, for example: 

```bash
Traceback (most recent call last): 
  File "learning_theano_train.py", line 25, in <module> 
    from lasagne.init import HeUniform, Constant 
... 
ImportError: cannot import name 'downsample' 
```

**Solution**: The issue is likely due to either Lasagne or Theano being out of date. While they are developed quickly in parallel, the latest stable versions may not always sync. The best bet is to pip install from the repo on GitHub to get the bleeding-edge versions of both: 

```bash
$ pip install --upgrade https://github.com/Theano/Theano/archive/master.zip 
$ pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip 
```
 
If using Anaconda, it should hijack pip (but not pip3) to install in its site-packages directory. If the install does not work, you can additionally try to clear the Theano cache in case any memory from an older version is inadvertently being stored: 

```bash
$ theano-cache purge 
```

**Error**: No module tqdm. 

```bash
ModuleNotFoundError: No module named 'tqdm' 
```
 
**Solution**: Install tqdm with either pip or conda. 

**Error**: Fail to import numpy multiarray extension. 

```bash
$ python basic.py  
ImportError: numpy.core.multiarray failed to import 
… 
$ python -c "import numpy;print(numpy.__version__);print(numpy.__file__)"; 
... 
ImportError: libopenblasp-r0-39a31c03.2.18.so: cannot open shared object file: No such file or directory 

During handling of the above exception, another exception occurred: 
… 
ImportError:  
Importing the multiarray numpy extension module failed.  Most 
likely you are trying to import a failed build of numpy. 
If you're working with a numpy git repo, try `git clean -xdf` (removes all 
files not under version control).  Otherwise reinstall numpy. 
```

**Solution**: Something funky with the new numpy installation is occurring. This may happen after installing required packages (including numpy) as part of a new environment in Anaconda. The root environment comes with numpy version 1.11, but the latest version (as of 5/23/17) is 1.12, which apparently is the cause of the bug. Uninstall numpy (scipy will be uninstalled as well) and reinstall numpy version 1.11 (the compatible scipy version will be installed automatically): 

```bash
$ conda uninstall numpy 
$ conda install numpy=1.11 scipy 
```
 
If this does not work, then try to reinstall numpy (and dependent packages) entirely. Using version 1.11 may also trigger the Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so error message above, leading to a cycle between the two errors as numpy is uninstalled and reinstalled with different versions. The package itself may be faulty, so clean all packages and cached items from Anaconda to do a fresh install: 

```bash
$ conda clean –a 
# say yes to uninstalling items 
$ conda install numpy=1.11 scipy scikit-image scikit-learn 
```
 
**Error**: scikit-image not installed. 

```bash
$ python learning_theano_train.py 
Traceback (most recent call last): 
  File "learning_theano_train.py", line 24, in <module> 
    import skimage.color, skimage.transform 
ModuleNotFoundError: No module named 'skimage' 
```

**Solution**: Install it! 

```bash
$ conda install scikit-image 
```
 
**Error**: A warning about changes in skimage default mode is displayed continuously during the program. 

```bash
warn("The default mode, 'constant', will be changed to 'reflect' in "/home/james/anaconda3/envs/vizdoom/lib/python3.6/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15. 
... 
```

**Solution**: This may be due to a newer version of scikit-image (which contains the skimage extension). Check the current version of scikit-image: 

```bash
$ conda list 
… 
scikit-image  0.13.0  np111py36_0 
… 
```
 
If the version is 0.13 or higher, then uninstall it and reinstall version 0.12: 

```bash
$ conda uninstall scikit-image 
$ conda install scikit-image=0.12 
```
 
Alternatively, include a warning filter in the code prior to use of scikit-image (probably during image preprocessing): 

```python
# Converts and downsamples the input image 
def preprocess(img): 
    img = skimage.transform.resize(img, resolution) 
    img = img.astype(np.float32) 
    return img 
... 
# Ignore skimage warning about change in default mode 
with warnings.catch_warnings(): 
    warnings.simplefilter("ignore") 
    state = preprocess(game.get_state().screen_buffer) 
```
 
**Error**: Decoding error using python pickle. 

```bash
File "testing_theano.py", line 135, in <module> 
    params = pickle.load(open(params_file_path, "rb")) 
UnicodeDecodeError: 'ascii' codec can't decode byte 0xb2 in position 2: ordinal not in range(128) 
```
 
**Solution**: This may be due to an incompatibility between Python 2 and 3. If the parameters were encoded in Python 2 and attempted to be loaded again in Python 3, then it will try to load a "binstring" object, which is ASCII, instead of the binary. Reformat the code for this particular instance by adding the argument encoding="latin1": 

```python
params = pickle.load(open(params_file_path, "rb"), encoding="latin1") 
```
 
**Error**: Config loader cannot add position (and other game variables) to game state. 

```bash
$ python test_theano.py <args> 
Initializing doom... 
WARNING! Loading config from: "../config/smaze_1.cfg". Unsupported value in lines 39-42: position_x. Lines ignored. 
```
 
**Solution**: Position (and some other game variables) were initially not accessible but later became supported. Unfortunately, these were not included in the config loader (src/lib/ViZDoomConfigLoader.cpp), so while these variables can be added in programs (`game.add_available_game_variable(<GameVariable>)`), they cannot be loaded from a config file. Simply add keys to code: 

```c
GameVariable ConfigLoader::stringToGameVariable(std::string str) { 
    if (str == "killcount") return KILLCOUNT; 
    if (str == "itemcount") return ITEMCOUNT; 
    … 
    // Add newly supported variables here 
    if (str == "position_x") return POSITION_X; 
    if (str == "position_y") return POSITION_Y; 
    if (str == "position_z") return POSITION_Z; 
} 
```

and recompile ViZDoom. The error message should disappear. 

**Error**: glibc outputs annoyingly long backtrace and memory map from memory corruption during python error. 

```bash
$ python learn_theano.py 
... 
*** Error in `python': free(): invalid pointer: 0x00007fe602d5b060 *** 
======= Backtrace: ========= 
/lib/x86_64-linux-gnu/libc.so.6(+0x777e5)[0x7fe6331cc7e5] 
/lib/x86_64-linux-gnu/libc.so.6(+0x7fe0a)[0x7fe6331d4e0a] 
… 
======= Memory map: ======== 
00400000-00401000 r-xp 00000000 08:02 19534490                           /home/james/anaconda3/envs/vizdoom/bin/python3.6 
00601000-00602000 rw-p 00001000 08:02 19534490                           /home/james/anaconda3/envs/vizdoom/bin/python3.6 
... 
```

**Solution**: Set environment variable `MALLOC_CHECK_` to 0. 

```bash
$ export MALLOC_CHECK_=0 
```
 
**Error (OS X)**: During building, CMake cannot find Boost target libraries. 

```bash
CMake Warning at /Applications/CMake.app/Contents/share/cmake-3.7/Modules/FindBoost.cmake:761 (message): 
  Imported targets not available for Boost version 106400 
```
 
**Solution**: The CMake version is too old. Boost 1.63 requires CMake 7 and up, and Boost 1.64 require CMake 8 and up. 

**Error**: GSettings cannot be saved: 

```bash
GLib-GIO-Message: Using the 'memory' GSettings backend.  Your settings will not be saved or shared with other applications. 
```

**Solution**: The dconf settings are likely pointing to Anaconda libraries rather than system libraries. 

```bash
$ ldd /usr/lib/x86_64-linux-gnu/gio/modules/libdconfsettings.so 
linux-vdso.so.1 =>  (0x00007ffe78d77000) 
    libgio-2.0.so.0 => /path_to_conda/anaconda3/envs/vizdoom/lib/libgio-2.0.so.0 (0x00007f1bbdb81000) 
    libgobject-2.0.so.0 => /path_to_conda/anaconda3/envs/vizdoom/lib/libgobject-2.0.so.0 (0x00007f1bbd932000) 
   ... 
```
 
Remove the path to the Anaconda libraries from the dconf settings by altering LD_LIBRARY_PATH: 

```bash
# cuda folder present if GPU is configured 
$ echo $LD_LIBRARY_PATH 
/path_to_conda/anaconda3/envs/vizdoom/lib:/usr/local/cuda/lib64 
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64 
$ echo $LD_LIBRARY_PATH 
/usr/local/cuda/lib64 
$ ldd /usr/lib/x86_64-linux-gnu/gio/modules/libdconfsettings.so 
linux-vdso.so.1 =>  (0x00007ffddc305000) 
    libgio-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 (0x00007f57f7e10000) 
    libgobject-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0 (0x00007f57f7bbd000) 
    ... 
```

If the problem resides only in a virtual env, for a more permanent fix look in the activate script under /path_to_conda/anaconda3/envs/env_name/etc/conda/activate.d. If pygpu is installed, then the script pygpu_vars.sh may be altering LD_LIBRARY_PATH. Delete this command to prevent adding the virtual env to the PATH.















## Useful Resources
ViZDoom setup: download and build working ViZDoom 
- [ViZDoom tutorial](http://vizdoom.cs.put.edu.pl/tutorial) 
- [ViZDoom GitHub](https://github.com/mwydmuch/ViZDoom) 

Slade3 setup (WAD editor): download Slade3 and ACS compiler 
- Slade3 map building tutorials 
    - [Slade tutorial ](http://slade.mancubus.net/index.php?page=wiki&wikipage=Part-1%3A-Creating-a-Map)
    - [Building a doom level](https://eev.ee/blog/2015/12/19/you-should-make-a-doom-level-part-1/)
- ACS language resources 
    - [ZDoom wiki](https://zdoom.org/wiki/)
    - [ACS overview](https://zdoom.org/wiki/ACS) and [beginner's guide](https://zdoom.org/wiki/A_quick_beginner%27s_guide_to_ACS)
    - [DECORATE overview](https://zdoom.org/wiki/DECORATE) 
- Other resources: 
    - [Doom editing tutorials](https://www.doomworld.com/forum/53-editing-tutorials/) (including this [beginner's guide](https://www.doomworld.com/vb/doom-editing-tutorials/55372-my-unfinished-newbie-udmf-level-editing-guide/) and [these chapters](http://maniacsvault.net/editing/))
    - [Sprite tutorial](http://web.archive.org/web/20081019052250/http://www.phobus.servegame.com:80/SpriteTutorialMain.html)

Neural network package installation
- [TensorFlow](https://www.tensorflow.org/install/install_linux) 
- [Theano](http://deeplearning.net/software/theano/install_ubuntu.html) 
- [Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) 

Additional resources 
- Anaconda 
- Visual Studio Code 
- GitHub 
    - Install GitHub (https://help.github.com/categories/bootcamp/) 
    - Learn basic git management (https://git-scm.com/docs/gittutorial) 