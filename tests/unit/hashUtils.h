#pragma once

// Helper for comparing object hashes
#define ASSERT_HASH_EQ(A, B, UPDATE_FN)                 \
    do {                                                \
        boost::uuids::detail::sha1 a;                   \
        boost::uuids::detail::sha1 b;                   \
        (A)->UPDATE_FN(a);                              \
        (A)->UPDATE_FN(b);                              \
        ASSERT_TRUE(a.get_digest() == b.get_digest());  \
    } while(false) 

#define ASSERT_HASH_NE(A, B, UPDATE_FN)                 \
    do {                                                \
        boost::uuids::detail::sha1 a;                   \
        boost::uuids::detail::sha1 b;                   \
        (A)->UPDATE_FN(a);                              \
        (B)->UPDATE_FN(b);                              \
        ASSERT_FALSE(a.get_digest() == b.get_digest()); \
    } while(false) 
