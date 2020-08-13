#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "DataModels.cu"
#include "VortexIndicator.cu"

using MinuteList = std::vector<MinuteData>;
using SymbolMinuteListMap = std::map<std::string, MinuteList>;

double *device_arr1;
double *device_outval;
double *error_val;
std::vector<std::string> dates;
std::vector<std::string> times;

bool IsPathExist(const std::string &s)
{
    struct stat buffer;
    return (stat (s.c_str(), &buffer) == 0);
}

std::string GetDirectory(std::string filename)
{
    std::string programPath(filename);
    size_t pos = programPath.rfind('/');
    if (pos == programPath.npos)
    {
        std::cerr << "Could get directory." << std::endl;
        return "";
    }
    return programPath.substr(0, pos);    
}


template<typename T>
std::set<T> ReadSymbolList(std::string filename)
{
    std::set<T> symbolsSet;

    std::ifstream stream(filename);
    if (stream.is_open())
    {
        std::string line;
        while (std::getline(stream, line))
        {
            // remove white space at end of each line
            T value;
            std::stringstream ss;
            ss << line;
            ss >> value;
            symbolsSet.insert(value);
        }
        stream.close();
    }
    else
        std::cerr << "Could not ready " << filename << std::endl;
    return symbolsSet;
}


MinuteList ReadMinutesFromFile(const std::string& filename)
{
    MinuteList minutes;
    std::ifstream stream;
    stream.open(filename);

    if(stream.is_open())
    {
        MinuteData mdata;
        while (stream >> mdata.date >> mdata.time >> mdata.open >> 
            mdata.high >> mdata.low >> mdata.close >> mdata.volume)
        {
            minutes.emplace_back(mdata);
        }
    }
    else
    {
        std::cout << "Error reading "  << filename << std::endl;
    }
    
    return std::move(minutes);
}


void WriteOutputToFile(std::string symbol, size_t symbolIdx, 
    size_t arraySize, size_t dataLength, int columns, 
    const std::vector<std::string>& dates, const std::vector<std::string>& times, 
    const double* device_outval, const std::string& outdir)
{
    std::ofstream stream;
    std::stringstream ssfile;
    ssfile << outdir << "/" << symbol << "-vortex.txt";

    stream.open(ssfile.str());
    
    if(stream.is_open())
    {
        //std::stringstream ss;
        size_t symbolOffset = symbolIdx * dataLength;

        for (size_t i = 0; i < dataLength; i++) 
        {
            size_t index = symbolOffset + i;
            size_t cIdx = (symbolOffset + i) * columns;
            
            stream << dates[index] << "\t"
                << times[index] << "\t"
                << device_outval[cIdx] << "\t"
                << device_outval[cIdx+1] << "\t"
                << device_outval[cIdx+2] << std::endl;
        }
        stream.close();
    }
    else
    {
        std::cerr << "Error opening file: " << ssfile.str() << std::endl;
    }
}


void FreeCudaVariables()
{
    if(device_arr1)
        cudaFree(device_arr1);
    if(device_outval)
        cudaFree(device_outval);
    if(error_val)
        cudaFree(error_val);
}


int main(int argc , char** argv)
{
    int symbolsCount = 0;
    int timesCount = 374;
    int columns = 3;
    int winSize = 14;
    SymbolMinuteListMap symbolMinuteMap;

    std::string programDir = GetDirectory(argv[0]);
    if(programDir.length() == 0)
        return 1;

    // get program current directory
    char fullpath[PATH_MAX]; 
    realpath(programDir.c_str(), fullpath); 
    programDir = std::string(fullpath);

    // Create output dir
    std::stringstream ssout;
    ssout << programDir << "/output";
    std::string OUTDIR = ssout.str();

    cudaFree(0);
    cudaError_t cudaStatus = cudaSetDevice(0);
    auto begin = std::chrono::steady_clock::now();

    if(argc < 2)
    {
        std::cerr << "Require input with minutes text files\n"
            << "Run: ./Vortex ./data winSize timesCount outdir";
        return 1;
    }
    if(argc >= 3)
    {
        winSize = std::stoi(argv[2]);
        std::cout << "Winsize: " << winSize << std::endl;
    }
    if(argc >= 4)
    {
        timesCount = std::stoi(argv[3]);
        std::cout << "Times: " << timesCount << std::endl;
    }
    if(argc >= 5)
    {
        OUTDIR = argv[4];
        std::cout << "Outdir: " << OUTDIR << std::endl;
    }

    if(!IsPathExist(OUTDIR))
    {
        const int dir_err = mkdir(OUTDIR.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {
            std::cerr <<"Error creating directory " << OUTDIR << std::endl;
            return 1;
        }
    }
    
    // read symbol list from file
    std::stringstream sspath;
    sspath << programDir << "/symbols.txt"; 
    auto symbolsSet = ReadSymbolList<std::string>(sspath.str());

    // read times list from file
    std::stringstream sstimes;
    sstimes << programDir << "/times.txt"; 
    auto timesSet = ReadSymbolList<int>(sstimes.str());
    auto timesArray = std::vector<int>(timesSet.begin(), timesSet.end());

    symbolsCount = symbolsSet.size();
    size_t dataLength;
    size_t arraySize;

    // read minute data for each symbol
    int k = 0;
    for(auto& symbol : symbolsSet)
    {
        std::stringstream ssfile;
        ssfile << argv[1] << "/" << symbol << ".txt";
        symbolMinuteMap[symbol] = ReadMinutesFromFile(ssfile.str());
        if(k == 0)
        {
            dataLength = symbolMinuteMap[symbol].size();
        }
        k++;
    }

    timesCount = timesArray.size();
    arraySize = symbolsCount * dataLength;

    if(arraySize <= 0)
    {
        std::cerr << "ArraySize is " << arraySize << std::endl;
        return 1;
    }
    
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaStatus = cudaMallocManaged(&device_arr1, arraySize * columns * sizeof(double));
    if (cudaStatus != cudaSuccess) 
    {
        std::cerr << "cudaMallocManaged device_arr1 " << cudaStatus << std::endl;
        return 1;
    }  

    cudaStatus = cudaMallocManaged(&device_outval, arraySize * columns * sizeof(double));
    if (cudaStatus != cudaSuccess) 
    {
        std::cerr << "cudaMallocManaged device_outval " 
            << cudaGetErrorString(cudaStatus) << std::endl;
        FreeCudaVariables();
        return 1;
    }  

    cudaStatus = cudaMallocManaged(&error_val, sizeof(double));
    if (cudaStatus != cudaSuccess) 
    {
        std::cerr << "cudaMallocManaged error_val " 
            << cudaGetErrorString(cudaStatus) << std::endl;
        FreeCudaVariables();  
        return 1;
    }

    error_val[0] = -999999;
    
    // initialize x and y arrays on the host
    size_t symbolIndex = 0;
    dates.resize(arraySize);
    times.resize(arraySize);
    
    for(auto& it : symbolMinuteMap)
    {
        int symbolOffset = symbolIndex * dataLength; 
        for (int i = 0; i < dataLength; i++) 
        {
            int index = symbolOffset+i;
            int cIdx = (symbolOffset+i) * columns;
    
            device_arr1[cIdx] = it.second[i].high;
            device_arr1[cIdx+1] = it.second[i].low;
            device_arr1[cIdx+2] = it.second[i].close;

            dates[index] = it.second[i].date;
            times[index] = it.second[i].time;

        }
        
        symbolIndex++;
    }

    std::cout << "added data\n";

    // compute number of blocks and threads
    int numThreads = 256; // Nvidia docs max num thread per block
    int blocks = std::ceil(1.0 * (arraySize + numThreads - 1) / numThreads);
    int bSize = std::ceil(std::sqrt(blocks));
    dim3 numBlocks(bSize, bSize);

    std::cout << "NumBlockX " << numBlocks.x 
        << " NumBlockY " << numBlocks.y << std::endl
        << "NumThreads " << (numBlocks.x * numBlocks.y * numThreads) << std::endl
        << "data size " << arraySize << std::endl;

    // call kernel
    VortexIndicator <<<numBlocks, numThreads>>> (
        device_arr1, symbolsCount, arraySize, dataLength, 
        timesCount, columns, device_outval, winSize, error_val);

    // Wait for GPU to finish before accessing on host
    cudaStatus = cudaDeviceSynchronize(); 
    if (cudaStatus != cudaSuccess) 
    {
        std::cerr << "Vortex Indicator kernel launch failed: " 
            << cudaGetErrorString(cudaStatus) << std::endl;
    }
    else
    {
        size_t symbolId = 0;
        for(auto& sit : symbolMinuteMap)
        {
            WriteOutputToFile(sit.first, symbolId, arraySize, dataLength, 
                columns, dates, times, device_outval, OUTDIR);
            symbolId++;
        }
    }
    
    // Free memory
    FreeCudaVariables();

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Vortex Indicator computed in " 
            << (elapsed/1000.0) << " senconds." << std::endl;

    cudaDeviceReset();
    return EXIT_SUCCESS;
}