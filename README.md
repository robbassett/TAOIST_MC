# TAOIST_MC
TrAnsmission Of IoniSing lightT - Monte Carlo: Simulated IGM UV Transmission

This code uses hydrogen absorption system statistics from the literature (redshift and column density distribution functions)
to simulate the transmission of ionizing radiation through Hydrogen in the IGM for sources at a specified redshift. 
The primary outputs of TAOIST-MC are ensembles of IGM transmission functions at UV wavelengths, i.e. T(lambda).

The main use case for TAOIST-MC to is produce Monte Carlo simulations of the expected observed ionising
flux from sources of ionizing radiation (star-forming galaxies or AGN) by coupling the IGM transmission functions produced
with model spectra of high redshift sources produced with population synthesis (e.g. BPASS). For example, using 10,000 IGM
sightlines from TAOIST-MC at z > 3.4 coupled to a BPASS model, one can estimate the detected flux in the photometric u-band
and compare with known sources from your favourite deep survey. In this way, for individual
detection of ionizing flux at high redshift, TAOIST-MC can be used to predict the probability distribution of ionising
photon escape in a more meaningful way than simply assuming a mean IGM transmission function for all galaxies, thus accounting
for possible biases for detections of ionizing radiation towards high transmission sightlines.

Development of TAOIST-MC is ongoing with plans to make the code more user friendly with a number of example script to be
provided. In the future, there are also plans to include Helium ionization, a process that may play an important role at 
higher photon energies.

Comments, questions, concerns? Email me at: rbassett.astro@gmail.com
