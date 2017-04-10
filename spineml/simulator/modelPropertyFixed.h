// SpineML simulator includes
#include "modelProperty.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelPropertyFixed
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
class ModelPropertyFixed : public ModelProperty
{
public:
    ModelPropertyFixed(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setValue(scalar value);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    scalar m_Value;
};
} // namespace SpineMLSimulator