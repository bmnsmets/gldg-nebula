#ifndef __PARTICLE_H_
#define __PARTICLE_H_

/// Represents a particle (currently, always an electron) in the simulation.
///
/// Aligned to 16 bytes, which pads it to 32 in single precision: the GPU then
/// loads and stores it with 128-bit accesses, each particle in one memory
/// sector. Files are read and written field by field, so their format does
/// not depend on this.
struct alignas(16) particle
{
	vec3 pos;        ///< Position
	vec3 dir;        ///< Direction (unnormalised).
	real kin_energy; ///< Kinetic energy
};

#endif // __PARTICLE_H_
