
namespace WeightUpdateModels
{
class Custom : public Base
{
  private:
    static WeightUpdateModels::Custom *s_Instance;
public:
    static const WeightUpdateModels::Custom *getInstance()
    {
        if(s_Instance == NULL)
        {
            s_Instance = new WeightUpdateModels::Custom;
        }
        return s_Instance;
    }
    typedef CustomValues::ParamValues ParamValues;
    typedef CustomValues::VarValues VarValues;

};
}
IMPLEMENT_MODEL(WeightUpdateModels::Custom);


