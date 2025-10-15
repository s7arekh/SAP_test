{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3.withPackages (ps: with ps; [
    ps.numpy
    ps.pandas
    ps.matplotlib
    ps.scikit-learn
    ps.tqdm
    ps.torch
    ps.plotly
    ps.nbformat
  ]);
in
  pkgs.mkShell {
    buildInputs = [ python ];
  }
