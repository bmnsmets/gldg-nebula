// Known-answer test for the Philox4x32-10 generator used on the GPU, with the
// test vectors of Random123 (kat_vectors). Exits with 1 on a mismatch.
#include "config/config.h"
#include "common/util/philox.h"

#include <iostream>
#include <iomanip>

int main()
{
	using nbl::util::philox_block;
	struct test_vector
	{
		philox_block counter;
		uint32_t key0, key1;
		philox_block expected;
	};
	const test_vector vectors[] = {
		{ { 0, 0, 0, 0 }, 0, 0,
		  { 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8 } },
		{ { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff }, 0xffffffff, 0xffffffff,
		  { 0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd } },
		{ { 0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344 }, 0xa4093822, 0x299f31d0,
		  { 0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1 } },
	};

	int failures = 0;
	for (const auto& v : vectors)
	{
		const philox_block out = nbl::util::philox4x32_10(v.counter, v.key0, v.key1);
		if (out.x != v.expected.x || out.y != v.expected.y
			|| out.z != v.expected.z || out.w != v.expected.w)
		{
			std::cerr << std::hex << "Philox4x32-10 mismatch for counter "
				<< v.counter.x << ' ' << v.counter.y << ' ' << v.counter.z << ' ' << v.counter.w
				<< ": got " << out.x << ' ' << out.y << ' ' << out.z << ' ' << out.w << '\n';
			++failures;
		}
	}

	// Uniform numbers lie in (0, 1]
	if (!(nbl::util::uniform_float(0) > 0 && nbl::util::uniform_float(0xffffffff) == 1
		&& nbl::util::uniform_double(0, 0) > 0 && nbl::util::uniform_double(0xffffffff, 0xffffffff) == 1))
	{
		std::cerr << "uniform conversion out of (0, 1]\n";
		++failures;
	}

	if (failures == 0)
		std::cout << "Philox4x32-10: all known-answer tests passed\n";
	return failures == 0 ? 0 : 1;
}
