For vmd setup:
1. In `Display -> Display Settings`, tune `Screen Hgt` for zooming in. Also, set `background` in `Display` to be white.
2. Add scale bar by  `Plugins -> Visualization -> Ruler`, select `scale` and change to orthographic view. The scale bar is automatically labeled in A but should be nm since the coordinates given by NERDSS PDB outputs are in nm.
3. Add water box in `Plugins -> TK console`:
```
pbc box -color black -width 5
```
4. Molecule visualizations:
4.1. `resname S and name COM`
- VDW, Color `ResName` and change the color settings to use magenta
- In TK console, `set sel [atomselect top "resname S and name COM"]` and then `$sel set radius 1`
4.2. `resname N and name COM`
- VDW, Color `ResName` and change the color settings to use green
- In TK console, `set sel [atomselect top "resname N and name COM"]` and then `$sel set radius 1`
4.3. `resname nuc and name COM`
- VDW, Color `ResName` and change the color settings to use gray
- In TK console, `set sel [atomselect top "resname nuc and name COM"]` and then `$sel set radius 5.5`
4.4. `resname P and name COM`
- VDW, Color `ResName` and change the color settings to use orange
- In TK console, `set sel [atomselect top "resname P and name COM"]` and then `$sel set radius 2`