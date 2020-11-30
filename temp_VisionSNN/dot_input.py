import numpy as np

###0---DEFINING THE STIMULUS-------------------------------------------------------------------------
def generate_dot(N_X, N_Y, N_frame,
                X_0=-1., Y_0=0.0, #initial position
                V_X=1., V_Y=0., #target speed
                dot_size=.05,
                blank_duration=0., blank_start=0.,
                flash_duration=0., flash_start=0., #flashing=False,
                width=2.,
                im_noise=0.05, #std of the background noise in images
                im_contrast=1.,
                NoisyTrajectory=False, traj_Xnoise=0, traj_Vnoise=0, reversal=False,
                hard=False, second_order=False, f=8, texture=False, sf_0=0.15,
                pink_noise=False,
                **kwargs):

    """

    >> pylab.imshow(concatenate((image[16,:,:],image[16,:,:]), axis=-1))

    """
    r_x = width / 2.
    r_y = r_x * N_Y / N_X
    x, y, t = np.mgrid[r_x*(-1+1./(N_X)):r_x*(1-1./(N_X)):1j*N_X,
                    r_y*(-1+1./(N_Y)):r_y*(1-1./(N_Y)):1j*N_Y,
                    0:(1-1./(N_frame+1)):1j*N_frame]

    if NoisyTrajectory:
       V_Y +=  traj_Vnoise/ np.float(N_frame) * np.random.randn(1, N_frame)
       V_X +=  traj_Vnoise/ np.float(N_frame) * np.random.randn(1, N_frame)

    x_ = np.amin([np.abs((x - X_0) - V_X*t*width), width - np.abs((x - X_0) - V_X*t*width)], axis=0)
    y_ = np.amin([np.abs((y - Y_0)- (V_Y )*t*width), width * N_Y / N_X - np.abs((y - Y_0) - V_Y*t*width)], axis=0)

    tube = np.exp(- (x_**2 + y_**2) /2. / dot_size**2) # size is relative to the width of the torus

    if hard : tube = (tube > np.exp(-1./2)) * 1.

    if texture:
        from experiment_cloud import generate as cloud
        texture = cloud(N_X, N_Y, N_frame, V_X=V_X, V_Y=V_Y, sf_0=sf_0, noise=0)
        texture /= np.abs(texture).max()
        tube *= texture
        #np.random.rand(N_X, N_Y, N_frame)

    if second_order:
        from experiment_fullgrating import generate as fullgrating
        tube *= fullgrating(N_X=N_X, N_Y=N_Y, N_frame=N_frame, width=width,
                            V_X=0., noise=0., f=f)

    tube /= tube.max()
    tube *= im_contrast

    # trimming
    if blank_duration>0.:
        N_start = np.floor(blank_start * N_frame)
        N_blank = np.floor(blank_duration * N_frame)
        tube[:, :, N_start:N_start + N_blank] = 0.

    if flash_duration>0.:
        N_start = int(flash_start * N_frame)
        N_flash = int(flash_duration * N_frame)
        # to have stationary flash, set speed to zero
        tube[:, :, :N_start] = 0.
        tube[:, :, (N_start + N_flash):] = 0.

    if reversal:
        # mirroring the second half period
        tube[:, :, N_frame//2:] = tube[::-1, :, N_frame//2:]

    # adding noise
    if pink_noise:
        from MotionClouds import get_grids, envelope_color, random_cloud, rectif
        fx, fy, ft = get_grids(N_X, N_Y, N_frame)
        envelope = envelope_color(fx, fy, ft, alpha=1) #V_X=V_X*N_Y/N_frame, V_Y=V_Y*N_X/N_frame, sf_0=sf_0, B_V=B_V, B_sf=B_sf, B_theta=B_theta)
        noise_image = 2*rectif(random_cloud(envelope))-1
#         print(noise_image.min(), noise_image.max())
        tube += im_noise * noise_image #np.random.randn(N_X, N_Y, N_frame)
    else:
        tube += im_noise * np.random.randn(N_X, N_Y, N_frame)

    return tube
