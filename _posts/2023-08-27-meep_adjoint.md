---
title:  "[FDTD] Optical Cavity / Waveguide Design With Meep Adjoint"
date:   2023-08-27 00:00:00 +0900
categories: jekyll update
layout: post
---

In this post, we are going to use some very simple but delicate tricks to design a 2D cavity with Meep python library. Meep provides robust and powerful FDTD tools and also comes with nice python wrappers for easier access.

The design will be based on free-form approach using adjoint method. The idea is simple; If you set the material as a lossy material, and FoM (Figure of Merit) as output gain, the optimizer will automatically try to minimize the optical path length and it will end up in a plain straight waveguide. However if you set the material as a positive gain material, the optimizer will do the opposite and a cavity with maximized path length will be an outcome.

Making a gain material in meep is pretty simple. More accurately, we are going to make it simple by calculating imaginary refractive indices. This can potentially make the numbers diverge in some designs but if that happens we can solve it by limiting the simulation time. The correct way would be using nonlinear materials with saturable gain, but for the demonstration here, it won't be necessary.

If we put the attenuation coefficient as α and flux gain per unit distance as gain,

α = ln(gain)

The refractive index n can be described as a real part and an imaginary part.

n = Re(n) + Im(n) * i = sqrt(εμ)

The imaginary part of the refractive index can be calculated from attenuation coefficient.

Im(n) = λα/4π

If we set the permeability μ as 1, the complex permittivity can be calculated from the complex refractive index

ε = n^2

Re(ε) = Re(n)^2 - Im(n)^2

Im(ε) = 2 * Re(n)^2 * Im(n)^2

And with these relations, we can control the gain of meep materials as follows.

{% highlight python %}
gain = 1.25
n_re, n_im = 1.5, -(1/fcen) * np.log(gain) / (4 * np.pi)
e_re, e_im = n_re**2 - n_im**2, 2 * n_re * n_im
d_conductivity = 2 * np.pi * fcen * e_im / e_re
gain_glass = mp.Medium(epsilon = e_re, D_conductivity = d_conductivity)
{% endhighlight %}

<br>![gain_guide](/assets/images/Meep/gain_guide.gif)<br>

When visualizing the Meep simulation fields, I used this colormap with transparency at 0 which provides a nice clarity.
{% highlight python %}
from matplotlib.colors import LinearSegmentedColormap
RdTrBu = LinearSegmentedColormap('RdTrBu', {colour : [[i/2, v[i], v[i]] for i in range(3)] for colour, v in {'red' : [0, 0, 1], 'green' : [0, 0, 0], 'blue' : [1, 0, 0], 'alpha' : [1, 0, 1]}.items()})
{% endhighlight %}

<br>![cmap_smaple](/assets/images/Meep/rbcmap.png)<br>


Now that we confirmed that the gain media work, we will design a cavity with these gain media.

{% highlight python %}
mp.verbosity(0)
resolution = 60        # pixels/μm

eta_e = 0.55
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
eta_i = 0.5
eta_d = 1 - eta_e
design_region_resolution = int(5 * resolution)

guide_width = 0.15
dpml = 0.25

sx = 3
sy = 1.5
cell_size = mp.Vector3(sx,sy,0)
pml_layers = [mp.PML(thickness = dpml)]

wvl_min = 0.532 - 0.025    # min wavelength
wvl_max = 0.534 + 0.025    # max wavelength
fmin = 1/wvl_max        # min frequency
fmax = 1/wvl_min        # max frequency
fcen = 0.5*(fmin+fmax)  # center frequency
df = fmax-fmin          # frequency width
frequencies = 1/np.linspace(wvl_min, wvl_max, 10)

src_pt = mp.Vector3(-0.5*sx+dpml + 0.2,0,0)

gain = 1.2
n_re, n_im = 4.2, -(1/fcen) * np.log(gain) / (4 * np.pi)
e_re, e_im = n_re**2 - n_im**2, 2 * n_re * n_im
d_conductivity = 2 * np.pi * fcen * e_im / e_re
gain_Si = mp.Medium(epsilon = e_re, D_conductivity = d_conductivity)

gain = 1.2
n_re, n_im = 1.5, -(1/fcen) * np.log(gain) / (4 * np.pi)
e_re, e_im = n_re**2 - n_im**2, 2 * n_re * n_im
d_conductivity = 2 * np.pi * fcen * e_im / e_re
gain_SiO2 = mp.Medium(epsilon = e_re, D_conductivity = d_conductivity)
{% endhighlight %}

And we set design region for adjoint methods.

{% highlight python %}
design_region_width = 1.6
design_region_height = 0.6
design_region_resolution = int(5 * resolution)

Nx = int(design_region_resolution * design_region_width) + 1
Ny = int(design_region_resolution * design_region_height) + 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), gain_SiO2, gain_Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(design_region_width, design_region_height, 0),
    ),
)
{% endhighlight %}

Before we start designing, we can measure the input flux for normalization purpose later by simulating a plain waveguide.

{% highlight python %}
# measure source flux for normalization
design_region.update_design_parameters(((X_g - X_g == 0) & (np.abs(Y_g) <= guide_width / 2)).ravel())
sim.reset_meep()
fl = sim.add_flux(frequencies, mp.FluxRegion(center = src_pt + mp.Vector3(0.1), size=mp.Vector3(y=sy - 2*dpml)))
sim.plot2D()
sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-5))
src_flux = mp.get_fluxes(fl)
src_power = np.abs(mp.get_fluxes(fl))**2
sim.reset_meep()
{% endhighlight %}

For transmittance (output port flux) objective function, we will simply use eigenmode coefficienct

{% highlight python %}
mode = 1
TE = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(
        center=mp.Vector3(x=sx / 2 - dpml - 0.1),
        size=mp.Vector3(y=0.75),
    ),
    mode,
)
{% endhighlight %}

But we also want to minimize the loss from scattering from the cavity. To calculate the scattering loss, we will calculate the flux out of the square box around the cavity.
{% highlight python %}
vr = mp.Volume(center=mp.Vector3(x = design_region_width / 2 + 0.1), size=mp.Vector3(y = design_region_height + 0.2)) # x
TFR = [mpa.FourierFields(sim, vr, component = c) for c in [mp.Ey, mp.Hz, mp.Ez, mp.Hy]]
vd = mp.Volume(center=mp.Vector3(y = -design_region_height / 2 - 0.1), size=mp.Vector3(x = design_region_width + 0.2)) # -y
TFD = [mpa.FourierFields(sim, vd, component = c) for c in [mp.Ex, mp.Hz, mp.Ez, mp.Hx]]
vl = mp.Volume(center=mp.Vector3(x = -design_region_width / 2 - 0.1), size=mp.Vector3(y = design_region_height + 0.2)) # -x
TFL = [mpa.FourierFields(sim, vl, component = c) for c in [mp.Ez, mp.Hy, mp.Ey, mp.Hz]]
vu = mp.Volume(center=mp.Vector3(y = design_region_height / 2 + 0.1), size=mp.Vector3(x = design_region_width + 0.2)) # y
TFU = [mpa.FourierFields(sim, vu, component = c) for c in [mp.Ez, mp.Hx, mp.Ex, mp.Hz]]
{% endhighlight %}

With these, we can design the FoM using all the fields

{% highlight python %}
ob_list = [TE, *TFR, *TFD, *TFL, *TFU]

def J(transmittance, *arg):
    gain = npa.abs(transmittance) ** 2 / src_flux
    loss_flux = sum([
        npa.sum(npa.real(npa.conjugate(c1)*c2 - npa.conjugate(c3)*c4) * sim.get_array_metadata(v)[3], axis = -1) for (c1, c2, c3, c4), v in zip((arg[i * 4 : i * 4 + 4] for i in range(4)), (vr, vd, vl, vu))
    ])
    loss_gain = loss_flux / src_flux # - gain
    return npa.mean(gain - 0.1 * loss_gain)

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-5,
)
{% endhighlight %}

<br>![design_region](/assets/images/Meep/design_region.png)<br>

Using nlopt.LD_MMA we optimize the cavity design
{% highlight python %}
algorithm = nlopt.LD_MMA
cur_beta = 4
beta_scale = 2
num_betas = 6
update_factor = 15
{% endhighlight %}
<br>![gain_cavity_png](/assets/images/Meep/gain_cavity.png)<br>

We animate the forward run to visulaize Ez field propagation.
{% highlight python %}
opt.step_funcs = [
    mp.at_beginning(mp.to_appended('eps', mp.output_epsilon)),
    mp.at_every(1/fcen/8*2, mp.to_appended('ez', mp.output_efield_z)),
]
opt([mapping(x, eta_i, cur_beta/2)], need_gradient = False)
opt.step_funcs = []
meta = opt.sim.get_array_metadata()

opt.plot2D(False)
plt.title(f'Optical cavity with gain')
plt.tight_layout()

with h5py.File('ez.h5') as f:
    lx, ly, lt = f['ez'].shape
    wave = plt.imshow(f['ez'][:,:,0].T, cmap = RdTrBu, vmin = -1.25, vmax = 1.25, extent = (lambda x:[x[0][0], x[0][-1], x[1][0], x[1][-1]])(meta))
    def update(frame):
        print(f'\rFrame {frame:04d} / {lt:04d}', end = '')
        wave.set_data(f['ez'][:,:,frame].T)
        return wave,
    ani = animation.FuncAnimation(fig = plt.gcf(), func = update, frames = range(int(lt * 0.4), int(lt * 0.6), 1), interval = 67, blit = True)
    ani.save('gain_cavity.gif')
plt.close()
{% endhighlight %}
<br>![gain_cavity_gif](/assets/images/Meep/gain_cavity.gif)<br>
The cavity seems to function well with minimal loss.