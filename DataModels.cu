#ifndef DATA_MODEL_CU
#define DATA_MODEL_CU

class MinuteData
{
public:
    std::string date;
    std::string time;
    double open;
    double high;
    double low;
    double close;
    double volume;
    
    bool operator < (const MinuteData& rhs) const
    {
        if(this->date < rhs.date)
            return true;
        if(this->date == rhs.date &&
            this->time < rhs.time)
        {
            return true;
        }
        return false;
    }
};

#endif