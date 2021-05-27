#pragma once

// Helper for comparing object hashes
#define ASSERT_MODEL_HASH_EQ(MODEL_A, MODEL_B)  \
    do {                                        \
        boost::uuids::detail::sha1 a;           \
        boost::uuids::detail::sha1 b;           \
        (MODEL_A)->updateHash(a);               \
        (MODEL_B)->updateHash(b);               \
        ASSERT_TRUE(Utils::hashesEqual(a, b));  \
    } while(false) 

#define ASSERT_MODEL_HASH_NE(MODEL_A, MODEL_B)  \
    do {                                        \
        boost::uuids::detail::sha1 a;           \
        boost::uuids::detail::sha1 b;           \
        (MODEL_A)->updateHash(a);               \
        (MODEL_B)->updateHash(b);               \
        ASSERT_FALSE(Utils::hashesEqual(a, b));  \
    } while(false) 
