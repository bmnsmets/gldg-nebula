#include <random>
#include "philox.h"

namespace nbl { namespace util {

namespace detail
{
	template<>
	class random_state<false>
	{
	public:
		using seed_type = typename std::mt19937::result_type;
		static constexpr seed_type default_seed = std::mt19937::default_seed;

		CPU random_state(seed_type seed = default_seed)
			: _generator(seed)
		{}

		PHYSICS real unit()
		{
#if CUDA_COMPILING
			// TODO: proper error message.
			return 0;
#else
			return std::generate_canonical<real, std::numeric_limits<real>::digits>(_generator);
#endif
		}

	private:
		std::mt19937 _generator;
	};

#if CUDA_COMPILER_AVAILABLE
	/**
	 * \brief GPU random numbers without stored state.
	 *
	 * The numbers are computed with Philox4x32-10 from the seed (key) and a
	 * counter made of the particle slot, the iteration, the kernel and the
	 * number of the draw. Each (slot, iteration, kernel) must be used by at
	 * most one thread, so that every counter is used once.
	 */
	template<>
	class random_state<true>
	{
	public:
		using seed_type = unsigned long long;
		static constexpr seed_type default_seed = 0;

		__device__ random_state(seed_type seed, uint32_t slot, uint32_t iteration, uint32_t kernel)
			: _key0(static_cast<uint32_t>(seed)),
			_key1(static_cast<uint32_t>(seed >> 32)),
			_counter{ 0, slot, iteration, kernel }
		{}

		PHYSICS real unit()
		{
#if USE_DOUBLE
			const uint32_t low = next();
			return uniform_double(low, next());
#else // USE_DOUBLE
			return uniform_float(next());
#endif // USE_DOUBLE
		}

	private:
		// Next 32 random bits; a new block of four every fourth call
		PHYSICS uint32_t next()
		{
			if (_used == 4)
			{
				_block = philox4x32_10(_counter, _key0, _key1);
				++_counter.x;
				_used = 0;
			}
			switch (_used++)
			{
				case 0: return _block.x;
				case 1: return _block.y;
				case 2: return _block.z;
				default: return _block.w;
			}
		}

		uint32_t _key0;
		uint32_t _key1;
		philox_block _counter;
		philox_block _block = { 0, 0, 0, 0 };
		int _used = 4;
	};

#endif // CUDA_COMPILER_AVAILABLE
} // namespace detail

}} // namespace nbl::util
