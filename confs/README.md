# Config Naming Convention

Configs are named as follows

	{implicit distribution}_{normal distribution}_{anisotropy type}_{background}

The options for each are outlined below

	implicit distribution:
		* gaussian
		* logistic
		* laplace

	normal distribution: 
		
		* delta
		* deltarelu
		* uniform
		* linearmixture
		* linearmixturerelu 
		* sggx

	anisotropy type:

		* constant 	(i.e. normal distribution ignores anisotropy value)	
		* spatial 	(i.e. learned spatial)
		* annealed 	(i.e. global value, annealed during training)
	
	background:

		* bg 	(i.e. background learned with NeRF++, used for BMVS)
		* white (constant white background, used for NeRF)
		* black (constant black background, used for DTU)
