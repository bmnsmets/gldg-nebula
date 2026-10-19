#ifndef __RANDOM_H_
#define __RANDOM_H_

namespace nbl { namespace util {

namespace detail
{
	/**
	 * \brief Holds the random-number generator, which is CPU or GPU specific
	 * (set by the template parameter). Should not be used directly.
	 */
	template<bool gpu_flag>
	class random_state;
}

/**
 * \brief Random number generator class.
 *
 * \tparam gpu_flag Set to true for a random number generator to be used in GPU
 *                  code, false to generate random numbers for the CPU.
 *
 * This class exposes functionality to generate random numbers from certain
 * distributions.
 */
template<bool gpu_flag>
class random_generator
	: public detail::random_state<gpu_flag>
{
public:
	using base_type = detail::random_state<gpu_flag>;
	using seed_type = typename base_type::seed_type;
	using base_type::default_seed;

	/**
	 * \brief CPU constructor.
	 */
	CPU random_generator(seed_type seed = default_seed)
		: base_type(seed)
	{
		static_assert(!gpu_flag,
			"Cannot call CPU constructor on a GPU random generator");
	}
	/**
	 * \brief GPU constructor.
	 *
	 * \param seed      Random seed.
	 * \param slot      Memory slot of the particle that draws the numbers.
	 * \param iteration Number of the simulation iteration.
	 * \param kernel    Number of the kernel within the iteration.
	 */
	GPU random_generator(seed_type seed, uint32_t slot, uint32_t iteration, uint32_t kernel)
		: base_type(seed, slot, iteration, kernel)
	{
		static_assert(gpu_flag,
			"Cannot call GPU constructor on a CPU random generator");
	}

	/**
	 * \brief Uniformly distributed between 0 and 1
	 */
	PHYSICS real unit()
	{
		return base_type::unit();
	}

	/**
	 * \brief Uniformly distributed between 0 and 2&pi;
	 */
	PHYSICS real phi()
	{
		return 2*pi*unit();
	}

	/**
	 * \brief Cosine and sine of an angle uniformly distributed between 0 and 2&pi;
	 */
	PHYSICS void unit_circle(real& cos_phi, real& sin_phi)
	{
		sincospir(2*unit(), &sin_phi, &cos_phi);
	}

	/**
	 * \brief Unit vector, uniformly distributed on the unit sphere
	 */
	PHYSICS vec3 uniform_vector()
	{
		const real cos_theta = 2*unit() - 1;
		real cos_phi, sin_phi;
		unit_circle(cos_phi, sin_phi);
		return make_unit_vec(cos_theta, cos_phi, sin_phi);
	}

	/**
	 * \brief Unit vector normal to the unit vector \p dir, uniformly
	 *        distributed in the plane normal to \p dir
	 */
	PHYSICS vec3 normal_vector(vec3 dir)
	{
		real cos_phi, sin_phi;
		unit_circle(cos_phi, sin_phi);
		return make_normal_vec(dir, cos_phi, sin_phi);
	}

	/**
	 * \brief Exponential, with typical constant tau (tau has units of return value)
	 */
	PHYSICS real exponential(real tau)
	{
		return -tau*logr(unit());
	}
};

}} // namespace nbl::util

#include "random.inl"

#endif // __RANDOM_H_
