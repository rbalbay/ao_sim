import numpy as np
import matplotlib.pyplot as plt
import datetime
from hcipy import *

wavelength = 1650e-9  # m
# wavelength = 600.0E-9  # m
zero_magnitude_flux = 3E10  # photon / s
stellar_magnitude = 10

r0 = 0.2
L0 = 20

# cn2 = [height(m), wind_velocity(m/s), cn2]
cn2 = [[9.57, 9, 649], [10.83, 9, 1299], [4.37, 15, 2598], [6.58, 25, 5196], [3.71, 40, 10392], [6.71, 21, 20785]]

pupil_pixels = 32  # 256
q = 4  # 16
num_airy = 8  # 32
super_sample_num = 10

m1_diameter = 3.940  # meter
m2_diameter = 0.975  # meter
spiders_width = 0.015  # 15 mm
spiders = 4
spatial_resolution = wavelength / m1_diameter   # pitch

dm_actuators = 22
dm_delta = 1E-3  # 1kHz
dm_leakage = 0.01  # leaky integrator
dm_gain = 0.5  # control loop gain
dm_rcond = 1E-2  # condition

wfs_integration_time = 1E-3  # wfs integration time

pupil_grid = make_pupil_grid(pupil_pixels, 4.1)
focal_grid = make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)

propagator = FraunhoferPropagator(pupil_grid, focal_grid)

dag_aperture = evaluate_supersampled(
    make_obstructed_circular_aperture(
        m1_diameter, (m2_diameter / m1_diameter), spiders, spiders_width),
    pupil_grid,
    super_sample_num)

actuator_spacing = m1_diameter / dm_actuators
dm_influence_functions = make_gaussian_influence_functions(pupil_grid, dm_actuators, actuator_spacing)
dm = DeformableMirror(dm_influence_functions)
dm_num_modes = dm.num_actuators

pwfs = PyramidWavefrontSensorOptics(pupil_grid, wavelength_0=wavelength)
camera = NoiselessDetector()

wf = Wavefront(dag_aperture, wavelength)
wf.total_power = 1

camera.integrate(pwfs.forward(wf), 1)
image_ref = camera.read_out()
image_ref /= image_ref.sum()

probe_amp = 0.01 * wavelength
slopes = []

for ind in range(dm_num_modes):
    if ind % 10 == 0:
        print("Slopes: [{:d} / {:d}]".format(ind + 1, dm_num_modes))
    slope = 0
    for s in [1, -1]:
        amp = np.zeros((dm_num_modes,))
        amp[ind] = s * probe_amp
        dm.actuators = amp
        dm_wf = dm.forward(wf)
        wfs_wf = pwfs.forward(dm_wf)
        camera.integrate(wfs_wf, 1)
        image = camera.read_out()
        image /= np.sum(image)
        slope += s * (image - image_ref) / (2 * probe_amp)
    slopes.append(slope)
slopes = ModeBasis(slopes)

reconstruction_matrix = inverse_tikhonov(slopes.transformation_matrix, rcond=dm_rcond, svd=None)
dm.random(0.1 * wavelength)

wf.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5) * wfs_integration_time

psf_in = propagator.forward(dm.forward(wf)).power / propagator.forward(wf).power.max()
input_strehl = psf_in.max()

psf = propagator.forward(wf).power
psf /= psf.max()

cn_squared = Cn_squared_from_fried_parameter(r0, 500e-9)

atm_layers = []
for i in cn2:
    layer = InfiniteAtmosphericLayer(pupil_grid, i[0], L0, i[1], i[2])
    atm_layers.append(layer)

atmosphere = MultiLayerAtmosphere(layers=atm_layers)
atmosphere.Cn_squared = cn_squared
atmosphere.reset()
atmosphere.evolve_until(1)

fig = plt.figure()
plt.switch_backend('QT5Agg')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

ao_strehl = ''
psf_tur = ''
phase = ''
wfs_image = ''
sci_cam_psf = ''
strehl_array = []

current_time = datetime.datetime.now()
frames_folder = current_time.strftime("%Y_%m_%d_%H_%M_%S")
# video = GifWriter("C:\\dag_ao\\" + frames_folder)

for i in np.linspace(0, 1, 101):
    print(f"Progress: [{i} / 100]")

    if stellar_magnitude < 20:
        stellar_magnitude += 1

    if stellar_magnitude > 0:
        wf.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5) * wfs_integration_time

    atmosphere.evolve_until(i)

    for j in np.linspace(0, dm_delta):
        wf_atm = atmosphere.forward(wf)
        wf_atm_psf = propagator.forward(wf_atm)
        psf_tur = wf_atm_psf.power
        psf_tur /= psf_tur.max()

        wf_dm = dm.forward(wf_atm)
        wf_pywfs = pwfs.forward(wf_dm)
        camera.integrate(wf_pywfs, 1)
        sci_cam = camera.read_out()

        wfs_image = large_poisson(sci_cam).astype(np.float)
        wfs_image /= np.sum(wfs_image)
        diff_image = wfs_image - image_ref

        dm.actuators = (1 - dm_leakage) * dm.actuators - dm_gain * reconstruction_matrix.dot(diff_image)
        phase = dag_aperture * dm.surface
        phase -= np.mean(phase[dag_aperture > 0])

        sci_cam_psf = propagator.forward(wf_dm).power
        sci_cam_psf /= sci_cam_psf.max()
        ao_strehl = get_strehl_from_pupil(psf, sci_cam_psf)

    strehl_array.append(ao_strehl)

    plt.clf()

    plt.subplot(2, 4, 1)
    plt.title(f'PSF (λ={wavelength}, S_mag={stellar_magnitude:.2f})')
    imshow_field(np.log10(psf), vmin=-5, cmap='gray', grid_units=spatial_resolution)

    plt.subplot(2, 4, 2)
    plt.title(f'Turbulence wf (r0={r0}, L0={L0})')
    imshow_field(atmosphere.phase_for(wavelength), pupil_grid, cmap='RdBu')

    plt.subplot(2, 4, 3)
    plt.title('PSF after turbulence')
    imshow_field(np.log10(psf_tur), vmin=-4, cmap='gray', grid_units=spatial_resolution)

    plt.subplot(2, 4, 4)
    plt.title(f'Telescope (m1={m1_diameter}m, m2={m2_diameter}m)')
    imshow_field(dag_aperture, cmap='gray')

    plt.subplot(2, 4, 5)
    plt.title('DM shape (22x22, 1kHz)')
    imshow_field(phase, vmin=-1e-6, vmax=1e-6, cmap='bwr_r')

    plt.subplot(2, 4, 6)
    plt.title('Pyramid WFS Image')
    imshow_field(wfs_image, cmap='gray')

    plt.subplot(2, 4, 7)
    plt.title('Science camera PSF')
    imshow_field(np.log10(sci_cam_psf), vmin=-5, cmap='gray', grid_units=spatial_resolution)

    plt.subplot(2, 4, 8)
    plt.title(f"Strehl ratio: {strehl_array[-1]:.2f}")
    plt.ylim(0, 1)
    plt.plot(strehl_array)

    plt.draw()
    plt.pause(0.001)
    # video.add_frame()
