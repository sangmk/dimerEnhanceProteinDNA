# Running nerdss may require dependencies to be installed
# Please follow the debug info to install the dependencies
# If the excecutable fails, please install nerdss from ...

# Run the NERDSS model to reproduce data
# PDB data used to plot figures are stored in the subfolder PDB.

# for figure 1
cd NERDSS_kpp8kt_tauPN00001
nohup ./nerdss -f parms.inp -c fixCoordinates.inp -s 359656438 > OUTPUT &
cd ..
# # for figure 2
cd NERDSS_kpp18kt_tauPN01
nohup ./nerdss -f parms.inp -c fixCoordinates.inp -s 196045226 > OUTPUT &
cd ..
# for figure 3
cd NERDSS_kpp18kt_tauPN00001
nohup ./nerdss -f parms.inp -c fixCoordinates.inp -s 284577757 > OUTPUT &
cd ..
# for figure 4
cd NERDSS_kpp18kt_tauPN10
nohup ./nerdss -f parms.inp -c fixCoordinates.inp -s 222842093 > OUTPUT &
cd ..
