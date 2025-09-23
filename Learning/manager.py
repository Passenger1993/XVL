import defusion,crack,clusters,empty

def generate_major_set(path = "C:/Users/Алексей/Documents/XVL_2025_set"):
	defusion.generate_dataset(f"{path}/defusion_set", num_images=10000)
	crack.generate_dataset(f"{path}/crack_set", num_images=10000)
	clusters.generate_dataset(f"{path}/clusters_set", num_images=10000, num_pores = 32)
	clusters.generate_dataset(f"{path}/single_pore_set", num_images=10000, num_pores = 1)
	empty.generate_dataset(f"{path}/empty_set", num_images=10000)

generate_major_set()