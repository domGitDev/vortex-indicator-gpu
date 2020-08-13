The Vortex Indicator (VTX) can be used to identify the start of a trend and subsequently affirm trend direction. First, a simple cross of the two oscillators can be used to signal the start of a trend. After this crossover, the trend is up when +VI is above -VI and down when -VI is greater than +VI. Second, a cross above or below a particular level can signal the start of a trend and these levels can be used to affirm trend direction. 

# COMPILE AND RUN

- Install Cuda ToolKit (cuda 9 and above)

- Add cuda lib to PATH and LD_LIBRARY_PATH
    - eg LINUX: 
        - export PATH=/usr/local/cuda/bin:$PATH
        - export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

- git clone https://github.com/domGitDev/vortex-indicator-gpu.git

- cd vortex-indicator-gpu

- ./compile.sh

- ./Vortex ./data 14

# INPUTS
    - symbols.txt (symbol name per line)
    - data (contains data per symbol in format [date, time, open, high, low, volume])
    - times.txt (this values depends on timeframe choosen)

# DATA ALIGNMENT
    - In order for the kelner work for multi symbols at once, data points count for all symbols should match.
    