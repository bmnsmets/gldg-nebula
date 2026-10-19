namespace nbl { namespace drivers {

namespace kernels
{
	using random_seed_t = typename util::random_generator<true>::seed_type;

	// Kernel numbers, which make the random numbers of each kernel differ
	enum kernel_id : uint32_t
	{
		INIT_KERNEL,
		INTERSECT_KERNEL,
		ELASTIC_KERNEL,
		INELASTIC_KERNEL
	};

	__global__ void init_buffer_data(bool* buffer_data, size_t N);

	template<typename particle_manager_t>
	__global__ void push_buffer_in(
		particle_manager_t particles,
		bool* buffer_in_data,
		particle* buffer_in_particles,
		uint32_t* buffer_in_tags,
		typename particle_manager_t::particle_index_t buffer_in_size);

	// Flag the detected particles and count the running ones (in counts[1])
	template<typename status_t>
	__global__ void flag_detected(status_t const* status, uint32_t capacity,
		status_t detected, status_t terminated, uint32_t* flags, uint32_t* counts);

	// Copy the flagged particles to position offset[i] of the output buffer and
	// terminate them; the number copied goes to counts[0]
	template<typename status_t>
	__global__ void gather_detected(status_t* status, particle const* particles,
		uint32_t const* tags, uint32_t capacity, status_t terminated,
		uint32_t const* flags, uint32_t const* offset,
		particle* out_particles, uint32_t* out_tags, uint32_t* counts);

	template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t>
	__global__ void init(particle_manager_t particles,
		material_manager_t materials,
		geometry_manager_t geometry,
		random_seed_t seed, uint32_t iteration,
		real min_energy, real max_energy);

	template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t, typename intersect_t>
	__global__ void intersect(particle_manager_t particles,
		material_manager_t materials,
		geometry_manager_t geometry,
		random_seed_t seed, uint32_t iteration,
		intersect_t isect);

	template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t>
	__global__ void inelastic(particle_manager_t particles,
		material_manager_t materials,
		geometry_manager_t geometry,
		random_seed_t seed, uint32_t iteration);

	template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t>
	__global__ void elastic(particle_manager_t particles,
		material_manager_t materials,
		geometry_manager_t geometry,
		random_seed_t seed, uint32_t iteration);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::gpu_driver(
	particle_index_t particle_capacity,
	intersect_t intersect,
	material_manager_t const & materials,
	geometry_manager_t const & geometry,
	real min_energy, real max_energy,
	seed_t seed
) :
	_min_energy(min_energy), _max_energy(max_energy),
	_particles(particle_manager_t::create(particle_capacity)),
	_materials(materials),
	_geometry(geometry),
	_intersect(intersect),
	_num_blocks(1 + particle_capacity/_threads_per_block),
	_seed(seed)
{
	/*
	 * Allocate output buffers
	 */
	cuda::cuda_new<particle>(&buffer_dout_particles, particle_capacity);
	cuda::cuda_new<uint32_t>(&buffer_dout_tags, particle_capacity);
	cuda::cuda_new<uint32_t>(&buffer_dout_flags, particle_capacity);
	cuda::cuda_new<uint32_t>(&buffer_dout_offset, particle_capacity);
	cuda::cuda_new<uint32_t>(&buffer_dout_counts, 2);
	cub::DeviceScan::ExclusiveSum(nullptr, buffer_scan_temp_size,
		buffer_dout_flags, buffer_dout_offset, particle_capacity);
	cuda::cuda_new<uint8_t>((uint8_t**)&buffer_scan_temp, buffer_scan_temp_size);

	cudaMallocHost(&buffer_hout_status, particle_capacity*sizeof(status_t));
	cudaMallocHost(&buffer_hout_particles, particle_capacity*sizeof(particle));
	cudaMallocHost(&buffer_hout_tags, particle_capacity*sizeof(uint32_t));
	cudaMallocHost(&buffer_hout_counts, 2*sizeof(uint32_t));

	// Fill device arrays with particle manager's default values.
	buffer_detected();

	// Kernels cannot run if the executable has no code for this GPU, or if the
	// driver is too old for the embedded PTX. Fail here instead of hanging later.
	cudaError_t err = cudaGetLastError();
	if (err == cudaSuccess)
		err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)
			+ " Check that this GPU's architecture is in CMAKE_CUDA_ARCHITECTURES"
			" and that the NVIDIA driver supports the CUDA version used to compile Nebula.");

	cudaStreamCreateWithFlags(&buffer_stream, cudaStreamNonBlocking);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::~gpu_driver()
{
	particle_manager_t::destroy(_particles);

	cudaFree(buffer_din_data);
	cudaFree(buffer_din_particles);
	cudaFree(buffer_din_tags);

	cudaFree(buffer_dout_particles);
	cudaFree(buffer_dout_tags);
	cudaFree(buffer_dout_flags);
	cudaFree(buffer_dout_offset);
	cudaFree(buffer_dout_counts);
	cudaFree(buffer_scan_temp);

	cudaFreeHost(buffer_hin_data);
	cudaFreeHost(buffer_hin_particles);
	cudaFreeHost(buffer_hin_tags);

	cudaFreeHost(buffer_hout_status);
	cudaFreeHost(buffer_hout_particles);
	cudaFreeHost(buffer_hout_tags);
	cudaFreeHost(buffer_hout_counts);

	cudaStreamDestroy(buffer_stream);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU auto gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::push(
	particle* particles,
	uint32_t* tags,
	particle_index_t N
) -> particle_index_t
{
	return _particles.push(particles, tags, N);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
template<typename detect_function>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::flush_detected(
	detect_function func)
{
	_particles.flush_detected(func);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU auto gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::get_running_count() const
-> particle_index_t
{
	return _particles.get_running_count();
}
template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU auto gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::get_detected_count() const
-> particle_index_t
{
	return _particles.get_detected_count();
}
template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU auto gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::get_running_and_detected_count() const
-> std::pair<particle_index_t, particle_index_t>
{
	const auto capacity = _particles._capacity;

	// Borrow the host output buffer, which is only used by flush_buffered()
	cudaMemcpy(buffer_hout_status, _particles._status,
		capacity*sizeof(status_t), cudaMemcpyDeviceToHost);

	particle_index_t N_running = 0;
	particle_index_t N_detected = 0;
	for (particle_index_t i = 0; i < capacity; ++i)
	{
		if (buffer_hout_status[i] == particle_manager_t::DETECTED)
			++N_detected;
		else if (buffer_hout_status[i] != particle_manager_t::TERMINATED)
			++N_running;
	}

	return { N_running, N_detected };
}


template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::allocate_input_buffers(
	particle_index_t N
)
{
	if (buffer_din_data != nullptr)
	{
		cudaFree(buffer_din_data);
		buffer_din_data = nullptr;
	}
	if (buffer_din_particles != nullptr)
	{
		cudaFree(buffer_din_particles);
		buffer_din_particles = nullptr;
	}
	if (buffer_din_tags != nullptr)
	{
		cudaFree(buffer_din_tags);
		buffer_din_tags = nullptr;
	}

	if (buffer_hin_data != nullptr)
	{
		cudaFreeHost(buffer_hin_data);
		buffer_hin_data = nullptr;
	}
	if (buffer_hin_particles != nullptr)
	{
		cudaFreeHost(buffer_hin_particles);
		buffer_hin_particles = nullptr;
	}
	if (buffer_hin_tags != nullptr)
	{
		cudaFreeHost(buffer_hin_tags);
		buffer_hin_tags = nullptr;
	}

	buffer_in_size = N;
	cuda::cuda_new<bool>(&buffer_din_data, N);
	cuda::cuda_new<particle>(&buffer_din_particles, N);
	cuda::cuda_new<uint32_t>(&buffer_din_tags, N);
	cudaMallocHost(&buffer_hin_data, N*sizeof(bool));
	cudaMallocHost(&buffer_hin_particles, N*sizeof(particle));
	cudaMallocHost(&buffer_hin_tags, N*sizeof(uint32_t));

	kernels::init_buffer_data<<<_num_blocks, _threads_per_block>>>(buffer_din_data, N);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::push_to_buffer(
	work_pool& pool
)
{
	// Copy device to host
	cudaMemcpyAsync(buffer_hin_data, buffer_din_data, buffer_in_size*sizeof(bool),
		cudaMemcpyDeviceToHost, buffer_stream);
	cudaMemcpyAsync(buffer_hin_particles, buffer_din_particles, buffer_in_size*sizeof(particle),
		cudaMemcpyDeviceToHost, buffer_stream);
	cudaMemcpyAsync(buffer_hin_tags, buffer_din_tags, buffer_in_size*sizeof(uint32_t),
		cudaMemcpyDeviceToHost, buffer_stream);
	cudaStreamSynchronize(buffer_stream);

	// Copy data
	pool.get_work(buffer_hin_data, buffer_hin_particles, buffer_hin_tags, buffer_in_size);

	// Copy back to device
	cudaMemcpyAsync(buffer_din_data, buffer_hin_data, buffer_in_size*sizeof(bool),
		cudaMemcpyHostToDevice, buffer_stream);
	cudaMemcpyAsync(buffer_din_particles, buffer_hin_particles, buffer_in_size*sizeof(particle),
		cudaMemcpyHostToDevice, buffer_stream);
	cudaMemcpyAsync(buffer_din_tags, buffer_hin_tags, buffer_in_size*sizeof(uint32_t),
		cudaMemcpyHostToDevice, buffer_stream);
	cudaStreamSynchronize(buffer_stream);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::push_to_simulation()
{
	_particles.sort();
	kernels::push_buffer_in<<<_num_blocks, _threads_per_block>>>(
		_particles, buffer_din_data, buffer_din_particles, buffer_din_tags, buffer_in_size);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::buffer_detected()
{
	// Gather the detected particles, in the order of their slots, with an
	// exclusive scan of their flags
	const auto capacity = _particles._capacity;
	cudaMemsetAsync(buffer_dout_counts, 0, 2*sizeof(uint32_t));
	kernels::flag_detected<<<_num_blocks, _threads_per_block>>>(
		_particles._status, capacity,
		status_t(particle_manager_t::DETECTED), status_t(particle_manager_t::TERMINATED),
		buffer_dout_flags, buffer_dout_counts);
	cub::DeviceScan::ExclusiveSum(buffer_scan_temp, buffer_scan_temp_size,
		buffer_dout_flags, buffer_dout_offset, capacity);
	kernels::gather_detected<<<_num_blocks, _threads_per_block>>>(
		_particles._status, _particles._particles, _particles._tags, capacity,
		status_t(particle_manager_t::TERMINATED),
		buffer_dout_flags, buffer_dout_offset,
		buffer_dout_particles, buffer_dout_tags, buffer_dout_counts);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
template<typename detect_function>
CPU auto gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::flush_buffered(
	detect_function function) -> particle_index_t
{
	// Copy the counts, then only the detected particles, device to host
	cudaMemcpyAsync(buffer_hout_counts, buffer_dout_counts, 2*sizeof(uint32_t),
		cudaMemcpyDeviceToHost, buffer_stream);
	cudaStreamSynchronize(buffer_stream);
	const particle_index_t N_detected = buffer_hout_counts[0];
	const particle_index_t N_running = buffer_hout_counts[1];

	cudaMemcpyAsync(buffer_hout_particles, buffer_dout_particles, N_detected*sizeof(particle),
		cudaMemcpyDeviceToHost, buffer_stream);
	cudaMemcpyAsync(buffer_hout_tags, buffer_dout_tags, N_detected*sizeof(uint32_t),
		cudaMemcpyDeviceToHost, buffer_stream);
	cudaStreamSynchronize(buffer_stream);

	for (particle_index_t i = 0; i < N_detected; ++i)
		function(buffer_hout_particles[i], buffer_hout_tags[i]);

	return N_running;
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::do_iteration()
{
	init();
	_particles.sort();
	events();
	++_iteration;
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::init()
{
	kernels::init<<<_num_blocks, _threads_per_block>>>(
		_particles, _materials, _geometry, _seed, _iteration,
		_min_energy, _max_energy
	);
}

template<typename scatter_list_t,
	typename intersect_t,
	typename geometry_manager_t
>
CPU void gpu_driver<scatter_list_t, intersect_t, geometry_manager_t>::events()
{
	kernels::intersect<<<_num_blocks, _threads_per_block>>>(
		_particles, _materials, _geometry, _seed, _iteration, _intersect
	);
	kernels::elastic<<<_num_blocks, _threads_per_block>>>(
		_particles, _materials, _geometry, _seed, _iteration
	);
	kernels::inelastic<<<_num_blocks, _threads_per_block>>>(
		_particles, _materials, _geometry, _seed, _iteration
	);
}

__global__ void kernels::init_buffer_data(bool* buffer_data, size_t N)
{
	const auto i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i < N)
		buffer_data[i] = 0;
}

template<typename particle_manager_t>
__global__ void kernels::push_buffer_in(
	particle_manager_t particles,
	bool* buffer_in_data,
	particle* buffer_in_particles,
	uint32_t* buffer_in_tags,
	typename particle_manager_t::particle_index_t buffer_in_size)
{
	const auto i = threadIdx.x+blockIdx.x*blockDim.x;
	if (i >= buffer_in_size || i >= particles.get_capacity())
		return;

	if (buffer_in_data[i] != 1)
		return;

	const auto particle_idx = particles.get_particle_index(particles.get_capacity() - i - 1);

	if (!particles.is_terminated(particle_idx))
		return;

	particles.add_particle(particle_idx, buffer_in_particles[i], buffer_in_tags[i]);
	buffer_in_data[i] = 0;
}

template<typename status_t>
__global__ void kernels::flag_detected(status_t const* status, uint32_t capacity,
	status_t detected, status_t terminated, uint32_t* flags, uint32_t* counts)
{
	const auto i = threadIdx.x + blockIdx.x*blockDim.x;
	bool running = false;
	if (i < capacity)
	{
		const status_t s = status[i];
		flags[i] = (s == detected);
		running = (s != detected && s != terminated);
	}

	// No thread may return before this barrier
	const int block_running = __syncthreads_count(running);
	if (threadIdx.x == 0 && block_running > 0)
		atomicAdd(&counts[1], uint32_t(block_running));
}

template<typename status_t>
__global__ void kernels::gather_detected(status_t* status, particle const* particles,
	uint32_t const* tags, uint32_t capacity, status_t terminated,
	uint32_t const* flags, uint32_t const* offset,
	particle* out_particles, uint32_t* out_tags, uint32_t* counts)
{
	const auto i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i >= capacity)
		return;

	if (flags[i])
	{
		out_particles[offset[i]] = particles[i];
		out_tags[offset[i]] = tags[i];
		status[i] = terminated;
	}
	if (i == capacity - 1)
		counts[0] = offset[i] + flags[i];
}

template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t>
__global__ void kernels::init(
	particle_manager_t particles,
	material_manager_t materials,
	geometry_manager_t geometry,
	random_seed_t seed, uint32_t iteration,
	real min_energy, real max_energy)
{
	const auto particle_idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(!particles.exists(particle_idx))
		return;

	// Ignore all particles that are not active
	if (!particles.active(particle_idx))
		return;

	// Get data from memory
	auto this_particle = particles[particle_idx];

	// If not in domain, terminate
	if (!geometry.in_domain(this_particle.pos))
	{
		particles.terminate(particle_idx);
		return;
	}

	// Next scattering event
	scatter_event next_scatter{
		0,
		geometry.get_max_extent()
	};

	// If not in a vacuum, get next scatter event
	auto this_material_idx = particles.get_material_index(particle_idx);
	if (materials.is_physical(this_material_idx))
	{
		const auto this_material = materials[this_material_idx];

		// Terminate if we are above or below the energy threshold
		// (which is with respect to the vacuum energy)
		if (this_particle.kin_energy < this_material.barrier + min_energy ||
			this_particle.kin_energy > this_material.barrier + max_energy)
		{
			particles.terminate(particle_idx);
			return;
		}

		// Sample next scattering event
		// TODO: case of no scattering events!
		util::random_generator<true> rng(seed, particle_idx, iteration, INIT_KERNEL);
		next_scatter = this_material.sample_path(this_particle, rng);
	}

	// Move particle to next event, unless there is a triangle in the way
	normalise(this_particle.dir);
	intersect_event next_intersect = geometry.propagate(
		this_particle.pos, this_particle.dir, next_scatter.distance,
		particles.get_last_triangle(particle_idx),
		particles.get_material_index(particle_idx)
	);

	if (next_intersect.isect_triangle == nullptr)
	{
		// No triangle intersection: move to scattering position.
		// Scatter there later (after sorting)
		this_particle.pos += this_particle.dir * next_scatter.distance;
		particles.set_scatter_event(particle_idx, next_scatter.type);
	}
	else
	{
		// Triangle intersection: move to triangle position.
		this_particle.pos += this_particle.dir*next_intersect.isect_distance;
		particles.set_intersect_event(particle_idx, next_intersect.isect_triangle);
	}

	// Store new particle data in global memory
	particles[particle_idx] = this_particle;
}

template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t, typename intersect_t>
__global__ void kernels::intersect(
	particle_manager_t particles,
	material_manager_t materials,
	geometry_manager_t geometry,
	random_seed_t seed, uint32_t iteration,
	intersect_t isect)
{
	// Get particle index
	const auto i = threadIdx.x+blockIdx.x*blockDim.x;
	if(!particles.exists(i))
		return;
	const auto particle_idx = particles.get_particle_index(i);

	// ignore all particles except those with an intersect event.
	if (!particles.next_intersect(particle_idx))
		return;

	util::random_generator<true> rng(seed, particle_idx, iteration, INTERSECT_KERNEL);
	isect.execute(materials, particles, particle_idx, rng);
}

template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t>
__global__ void kernels::inelastic(
	particle_manager_t particles,
	material_manager_t materials,
	geometry_manager_t geometry,
	random_seed_t seed, uint32_t iteration)
{
	// Get particle index
	const auto i = threadIdx.x+blockIdx.x*blockDim.x;
	if(!particles.exists(i))
		return;
	const auto particle_idx = particles.get_particle_index(i);

	// ignore all particles except those with an inelastic event.
	if (!particles.next_inelastic(particle_idx))
		return;

	// If we can't make a secondary right now, set to pending and wait an iteration.
	if (!particles.secondary_slot_free())
	{
		particles.pending(particle_idx);
		return;
	}

	// We are definitely scattering inelastically now (status needs to be set if previously pending)
	particles.inelastic(particle_idx);

	// forget last intersected triangle. This event might cause us to scatter back into that triangle
	// and we don't want to ignore that triangle if so.
	particles.forget_last_triangle(particle_idx);

	// TODO: event types hardcoded here
	util::random_generator<true> rng(seed, particle_idx, iteration, INELASTIC_KERNEL);
	materials[particles.get_material_index(particle_idx)].execute(1, particles, particle_idx, rng);
}

template<typename particle_manager_t, typename material_manager_t, typename geometry_manager_t>
__global__ void kernels::elastic(particle_manager_t particles,
	material_manager_t materials,
	geometry_manager_t geometry,
	random_seed_t seed, uint32_t iteration)
{
	// Get particle index
	const auto i = threadIdx.x+blockIdx.x*blockDim.x;
	if(!particles.exists(i))
		return;
	const auto particle_idx = particles.get_particle_index(i);

	// ignore all particles except those with an elastic event.
	if (!particles.next_elastic(particle_idx))
		return;

	// forget last intersected triangle. This event might cause us to scatter back into that triangle
	// and we don't want to ignore that triangle if so.
	particles.forget_last_triangle(particle_idx);

	// TODO: event types hardcoded here
	util::random_generator<true> rng(seed, particle_idx, iteration, ELASTIC_KERNEL);
	materials[particles.get_material_index(particle_idx)].execute(2, particles, particle_idx, rng);
}

}} // namespace nbl::drivers
