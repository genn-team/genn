
namespace PostsynapticModels
{
class Custom : public Base
{
  private:
    static PostsynapticModels::Custom *s_Instance;
public:
    static const PostsynapticModels::Custom *getInstance()
    {
        if(s_Instance == NULL)
        {
            s_Instance = new PostsynapticModels::Custom;
        }
        return s_Instance;
    }
    typedef CustomValues::ParamValues ParamValues;
    typedef CustomValues::VarValues VarValues;

    CustomValues::ParamValues* make_ParamValues( const std::vector< double > & vals )
    {
        return new CustomValues::ParamValues( vals );
    }
    
    CustomValues::VarValues* make_VarValues( const std::vector< double > & vals )
    {
        return new CustomValues::VarValues( vals );
    }
};
}
IMPLEMENT_MODEL(PostsynapticModels::Custom);


