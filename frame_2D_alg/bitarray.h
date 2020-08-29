/**
 * Bit array manipulation. These operations assume array elements of
 * type long.
 * Source:
 * http://www.mathcs.emory.edu/~cheung/Courses/255/Syllabus/1-C-intro/bit-array.html
 */

#include <string.h>

#define setbit(A, i) (A[i >> 5] |= (1 << (i & 0x1F)))
#define clearbit(A, i) (A[i >> 5] &= ~(1 << (i & 0x1F)))
#define testbit(A, i) (A[i >> 5] & (1 << (i & 0x1F)))
#define setbits(A, n) memset(A, 0xFF, (n/32 + 1) * sizeof(long))
#define clearbits(A, n) memset(A, 0x00, (n/32 + 1) * sizeof(long))