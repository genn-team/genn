/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_Runtime_ArrayBase = R"doc()doc";

static const char *__doc_Sensors_DVS = R"doc()doc";

static const char *__doc_Sensors_DVS_CropRect = R"doc(Rectangle struct used to)doc";

static const char *__doc_Sensors_DVS_CropRect_bottom = R"doc()doc";

static const char *__doc_Sensors_DVS_CropRect_left = R"doc()doc";

static const char *__doc_Sensors_DVS_CropRect_right = R"doc()doc";

static const char *__doc_Sensors_DVS_CropRect_top = R"doc()doc";

static const char *__doc_Sensors_DVS_DVS = R"doc()doc";

static const char *__doc_Sensors_DVS_Polarity = R"doc(How to handle event polarity)doc";

static const char *__doc_Sensors_DVS_Polarity_MERGE = R"doc(Merge together on and off events)doc";

static const char *__doc_Sensors_DVS_Polarity_OFF_ONLY = R"doc(Only process off events)doc";

static const char *__doc_Sensors_DVS_Polarity_ON_ONLY = R"doc(Only process on events)doc";

static const char *__doc_Sensors_DVS_Polarity_SEPERATE = R"doc(Process on and off events seperately)doc";

static const char *__doc_Sensors_DVS_create = R"doc(Create DVS interface for camera type)doc";

static const char *__doc_Sensors_DVS_getHeight = R"doc(Get vertical resolution of DVS)doc";

static const char *__doc_Sensors_DVS_getWidth = R"doc(Get horizontal resolution of DVS)doc";

static const char *__doc_Sensors_DVS_m_Device = R"doc()doc";

static const char *__doc_Sensors_DVS_m_Height = R"doc()doc";

static const char *__doc_Sensors_DVS_m_Width = R"doc()doc";

static const char *__doc_Sensors_DVS_readEvents = R"doc(Read all events received since last call to readEvents into array)doc";

static const char *__doc_Sensors_DVS_start = R"doc(Start streaming events from DVS)doc";

static const char *__doc_Sensors_DVS_stop = R"doc(Stop streaming events from DVS)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

