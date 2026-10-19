#ifndef __PHILOX_H_
#define __PHILOX_H_

#include <cstdint>

namespace nbl { namespace util {

/**
 * \brief Four 32-bit words: a Philox counter, or a block of random output.
 */
struct philox_block
{
	uint32_t x, y, z, w;
};

/**
 * \brief Philox4x32-10 counter-based random number generator.
 *
 * Maps a 128-bit counter and a 64-bit key to 128 random bits. Distinct
 * counters give independent outputs, so random numbers can be computed from
 * (key, counter) without storing a generator state. This is the generator of
 * Salmon et al., "Parallel random numbers: as easy as 1, 2, 3" (SC11), also
 * used by Random123 and curand.
 *
 * \param counter Counter, unique for every call with the same key.
 * \param key0    Low word of the key.
 * \param key1    High word of the key.
 */
inline PHYSICS philox_block philox4x32_10(philox_block counter, uint32_t key0, uint32_t key1)
{
	for (int round = 0; round < 10; ++round)
	{
		const uint64_t product0 = uint64_t(0xD2511F53u) * counter.x;
		const uint64_t product1 = uint64_t(0xCD9E8D57u) * counter.z;
		counter = {
			uint32_t(product1 >> 32) ^ counter.y ^ key0,
			uint32_t(product1),
			uint32_t(product0 >> 32) ^ counter.w ^ key1,
			uint32_t(product0)
		};
		key0 += 0x9E3779B9u;
		key1 += 0xBB67AE85u;
	}
	return counter;
}

/**
 * \brief Uniform float in (0, 1] from 32 random bits, using the top 24 bits.
 *
 * Never returns 0, like curand_uniform(), so that log() of it is finite.
 */
inline PHYSICS float uniform_float(uint32_t bits)
{
	return ((bits >> 8) + 1) * (1.f / 16777216.f);
}

/**
 * \brief Uniform double in (0, 1] from 64 random bits, using the top 53 bits.
 */
inline PHYSICS double uniform_double(uint32_t low, uint32_t high)
{
	const uint64_t bits = (uint64_t(high) << 32) | low;
	return ((bits >> 11) + 1) * (1. / 9007199254740992.);
}

}} // namespace nbl::util

#endif // __PHILOX_H_
