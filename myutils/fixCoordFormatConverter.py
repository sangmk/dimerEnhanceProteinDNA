#!/usr/bin/env python3
"""
Molecular Coordinates to PDB Format Converter

This script converts custom molecular coordinate files to standard PDB format.
The input format contains NEWTYPE sections with molecule types and coordinates.
The output PDB format assigns each molecule type to a separate chain and 
names all atoms as "COM" (center of mass).

Usage:
    python mol_to_pdb.py input_file.txt output_file.pdb

Input Format:
    NEWTYPE
    <Molecule type>
    <Number of molecules>
    <x>\t<y>\t<z>
    ...

Output: Standard PDB format file
"""

import sys
import argparse
from datetime import datetime
from typing import List, Tuple


class MolecularCoordinate:
    """Class to store molecular coordinate data"""
    
    def __init__(self, mol_type: str, x: float, y: float, z: float):
        self.mol_type = mol_type
        self.x = x
        self.y = y
        self.z = z


class MoleculeType:
    """Class to store a collection of molecules of the same type"""
    
    def __init__(self, name: str):
        self.name = name
        self.atoms = []
    
    def add_atom(self, x: float, y: float, z: float):
        """Add an atom coordinate to this molecule type"""
        self.atoms.append(MolecularCoordinate(self.name, x, y, z))


class MolToPDBConverter:
    """Main converter class"""
    
    def __init__(self):
        self.molecule_types = []
    
    def parse_input_file(self, filename: str) -> bool:
        """
        Parse the input molecular coordinate file
        
        Args:
            filename (str): Input file path
            
        Returns:
            bool: True if parsing successful, False otherwise
        """
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            line_index = 0
            while line_index < len(lines):
                line = lines[line_index].strip()
                
                if line == "NEWTYPE":
                    # Parse new molecule type section
                    line_index += 1
                    
                    # Get molecule type name
                    if line_index >= len(lines):
                        break
                    mol_type_name = lines[line_index].strip()
                    line_index += 1
                    
                    # Get number of molecules
                    if line_index >= len(lines):
                        break
                    try:
                        num_molecules = int(lines[line_index].strip())
                    except ValueError:
                        print(f"Error: Invalid number of molecules at line {line_index + 1}")
                        return False
                    line_index += 1
                    
                    # Create new molecule type
                    mol_type = MoleculeType(mol_type_name)
                    
                    # Read coordinates
                    for i in range(num_molecules):
                        if line_index >= len(lines):
                            print(f"Error: Unexpected end of file while reading coordinates")
                            return False
                        
                        coord_line = lines[line_index].strip()
                        line_index += 1
                        
                        # Parse coordinates (tab or space separated)
                        coords = coord_line.replace('\t', ' ').split()
                        if len(coords) < 3:
                            print(f"Error: Invalid coordinate format at line {line_index}")
                            return False
                        
                        try:
                            x = float(coords[0])
                            y = float(coords[1])
                            z = float(coords[2])
                            mol_type.add_atom(x, y, z)
                        except ValueError:
                            print(f"Error: Invalid coordinate values at line {line_index}")
                            return False
                    
                    self.molecule_types.append(mol_type)
                else:
                    line_index += 1
            
            print(f"Successfully parsed {len(self.molecule_types)} molecule types")
            for mol_type in self.molecule_types:
                print(f"  {mol_type.name}: {len(mol_type.atoms)} atoms")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Input file '{filename}' not found")
            return False
        except Exception as e:
            print(f"Error reading input file: {e}")
            return False
    
    def write_pdb_file(self, filename: str) -> bool:
        """
        Write the molecular data to PDB format
        
        Args:
            filename (str): Output PDB file path
            
        Returns:
            bool: True if writing successful, False otherwise
        """
        try:
            with open(filename, 'w') as file:
                # Write PDB header
                today = datetime.now().strftime("%d-%b-%y").upper()
                file.write(f"HEADER CONVERTED MOLECULE FILE                    {today}   NONE\n")
                
                atom_index = 1
                
                for mol_type in self.molecule_types:
                    residue_index = 1
                    
                    # Format residue name to 3 characters
                    residue_name = mol_type.name[:3].ljust(3)
                    
                    for atom in mol_type.atoms:
                        # Format PDB ATOM record according to PDB specification
                        atom_line = (
                            f"ATOM  "                                    # Record name (6 chars)
                            f"{atom_index:>5} "                         # Atom serial number (5 chars + space)
                            f"{'COM':<4}"                               # Atom name (4 chars)
                            f" "                                        # Alternate location (1 char)
                            f"{residue_name:<3}"                        # Residue name (3 chars)
                            f" "                                        # Space (1 char)
                            f"_"                               # Chain ID (1 char)
                            f"{residue_index:>4} "                      # Residue sequence number (4 chars + space)
                            f"   "                                      # Insertion code + spaces (3 chars)
                            f"{atom.x:>8.3f}"                          # X coordinate (8 chars)
                            f"{atom.y:>8.3f}"                          # Y coordinate (8 chars)
                            f"{atom.z:>8.3f}"                          # Z coordinate (8 chars)
                            f"{'1.00':>6}"                             # Occupancy (6 chars)
                            f"{'0.00':>6}"                             # Temperature factor (6 chars)
                            f"{' '*10}"                                # Spaces (10 chars)
                            f"{atom.mol_type[0]:>2}"                   # Element symbol (2 chars)
                        )
                        file.write(atom_line + "\n")
                        
                        atom_index += 1
                        residue_index += 1
                    
                    # Write TER record to mark end of chain
                    ter_line = (
                        f"TER   "                                      # Record name (6 chars)
                        f"{atom_index:>5}"                            # Atom serial number (5 chars)
                        f"      "                                     # Spaces (6 chars)
                        f"{residue_name:<3}"                          # Residue name (3 chars)
                        f" "                                          # Space (1 char)
                        f"_"                                 # Chain ID (1 char)
                        f"{residue_index-1:>4}"                       # Last residue number (4 chars)
                    )
                    file.write(ter_line + "\n")
                    
                    atom_index += 1
                
                # Write END record
                file.write("END\n")
            
            print(f"Successfully wrote PDB file: {filename}")
            return True
            
        except Exception as e:
            print(f"Error writing PDB file: {e}")
            return False
    
    def convert(self, input_file: str, output_file: str) -> bool:
        """
        Convert molecular coordinate file to PDB format
        
        Args:
            input_file (str): Input file path
            output_file (str): Output PDB file path
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        print(f"Converting {input_file} to {output_file}")
        
        if not self.parse_input_file(input_file):
            return False
        
        if not self.write_pdb_file(output_file):
            return False
        
        print("Conversion completed successfully!")
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert molecular coordinate files to PDB format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python mol_to_pdb.py coordinates.txt output.pdb
    python mol_to_pdb.py -i input.txt -o result.pdb

Input file format:
    NEWTYPE
    S
    4
    -27.50	0	0
    -25.50	0	0
    17.50	0	0
    35.00	0	0
    NEWTYPE
    N
    2
    -29.21	0	0
    -10.70	0	0
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Input molecular coordinate file')
    parser.add_argument('output_file', nargs='?', help='Output PDB file')
    parser.add_argument('-i', '--input', dest='input_alt', help='Input file (alternative syntax)')
    parser.add_argument('-o', '--output', dest='output_alt', help='Output file (alternative syntax)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine input and output files
    input_file = args.input_file or args.input_alt
    output_file = args.output_file or args.output_alt
    
    if not input_file or not output_file:
        parser.print_help()
        sys.exit(1)
    
    # Create converter and run conversion
    converter = MolToPDBConverter()
    success = converter.convert(input_file, output_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()