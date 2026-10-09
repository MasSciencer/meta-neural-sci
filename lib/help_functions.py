import os
import numpy as np
import torch
import pickle
import meep as mp
import pandas as pd
from typing import Union, Sequence


def get_gap_from_pinhole_to_surface(
        D_aperture: float,
        D_pinhole: float,
        max_angle_in_deg: float,
) -> float:

    max_angle_in_rad = np.radians(max_angle_in_deg)
    d_gap = (D_aperture - D_pinhole)/(2*np.tan(max_angle_in_rad))

    return d_gap

def get_max_angle_in_deg(D_aperture: float, D_pinhole: float, gap_from_pinhole_to_surface: float) -> float:
    max_angel = np.arctan ((D_aperture - D_pinhole) / (2 * gap_from_pinhole_to_surface))
    return np.rad2deg(max_angel)

def get_angle_list(n_angles: int, D_pinhole: float, gap_from_pinhole_to_surface: float):
    angle_list = []
    for i in range(n_angles):
        next_angle_in_rad = np.arctan ((i*D_pinhole) / gap_from_pinhole_to_surface)
        next_angle_in_deg = np.rad2deg(next_angle_in_rad)
        angle_list.append(next_angle_in_deg)
    return angle_list

def find_D_pinhole(
        n_angels: int,
        D_aperture: float,
) -> float:
    """
    Definition: Estimate the diaphragm diameter from the number of optimization angles.

    :param n_angels:
    :param D_aperture:
    :return:
    """
    if n_angels <= 1:
        return D_aperture
    else:
        angle_orders = 2 * n_angels - 1
        d_pinhole = D_aperture / angle_orders

    return d_pinhole

def save_params(
    params: Union[torch.Tensor, Sequence[torch.Tensor]],
    name: str,
    np_dir: str = ".",
    pt_dir: str = "."
):
    """
    Save a tensor or a sequence of tensors:

    - If params is a torch.Tensor, save it to name.npy and name.pt.
    - If params is a list or tuple of torch.Tensor objects, save the entire sequence
      to name.npy (as an object array) and name.pt.

    Args:
        params (torch.Tensor or Sequence[torch.Tensor]):
            Tensor or sequence of tensors to save.
        name (str): Base filename without an extension.
        np_dir (str): Directory for .npy files (current directory by default).
        pt_dir (str): Directory for .pt files (current directory by default).
    """
    os.makedirs(np_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)

    np_path = os.path.join(np_dir, f"{name}.npy")
    pt_path = os.path.join(pt_dir, f"{name}.pt")

    if isinstance(params, torch.Tensor):
        # --- Single tensor ---
        tensor_cpu = params.detach().cpu()
        np_array = tensor_cpu.numpy()
        np.save(np_path, np_array)
        torch.save(params, pt_path)
        print(f"[Tensor] Saved NumPy array to {np_path}")
        print(f"[Tensor] Saved Torch tensor to {pt_path}")

    elif isinstance(params, (list, tuple)):
        # --- Sequence of tensors ---
        if not all(isinstance(t, torch.Tensor) for t in params):
            raise ValueError("All sequence elements must be torch.Tensor objects")

        # Convert each tensor to NumPy.
        np_list = [t.detach().cpu().numpy() for t in params]
        # Use an object array to support tensors with different shapes.
        obj_array = np.array(np_list, dtype=object)
        np.save(np_path, obj_array, allow_pickle=True)

        # Save the original sequence of tensors.
        torch.save(params, pt_path)

        print(f"[List]   Saved list of {len(params)} NumPy arrays to {np_path}")
        print(f"[List]   Saved list of {len(params)} Torch tensors to {pt_path}")

    else:
        raise TypeError("params must be a torch.Tensor or a list/tuple of torch.Tensor")

def save_history(loc_history=None, save=True, index=None):
    """
    Save or load the loc_history dictionary.

    Parameters:
        loc_history (dict | None): Dictionary to save when save=True.
        save (bool): True saves the dictionary; False loads it.
        index (int | None): Optional filename suffix ("loc_history_angle_{index}.pkl").

    Returns:
        str | dict: Filename when saving; dictionary when loading.
    """
    filename = "loc_history.pkl" if index is None else f"loc_history_angle_{index}.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump(loc_history, f)
        print(f"[INFO] History saved to {filename}")
        return filename
    else:
        with open(filename, "rb") as f:
            loc_history = pickle.load(f)
        print(f"[INFO] History loaded from {filename}")
        return loc_history

def recenter_and_crop(whole_int: np.ndarray, abscissa: np.ndarray, ref_index: int):
    """
    Center and symmetrically crop IOptimized so the selected wavelength peak is at the center.
    Return the cropped array and its corresponding abscissa vector.

    Parameters
    ----------
    IOptimized : np.ndarray
        Array of shape (n, m), with n sample points and m wavelengths.
    abscissa : np.ndarray
        Vector of length n containing absolute sample coordinates (e.g., in microns).
    ref_index : int
        Wavelength column index used as the reference for centering.

    Returns
    -------
    (np.ndarray, np.ndarray)
        - Centered, cropped array of shape (n', m).
        - Abscissa vector of length n'.
    """
    n, m = whole_int.shape
    center = n // 2

    # Find the peak index for the reference wavelength.
    ref_max_idx = np.argmax(whole_int[:, ref_index])

    # Compute the shift.
    shift = center - ref_max_idx

    # Shift the data.
    shifted = np.roll(whole_int, shift, axis=0)
    abscissa_shifted = np.roll(abscissa, shift)

    # Determine how many points to crop from each side.
    cut = abs(shift)
    if cut > 0:
        shifted = shifted[cut:-cut, :]
        abscissa_shifted = abscissa_shifted[cut:-cut]
        print(f"[INFO] Centering: peak was at {ref_max_idx}, shift = {shift}. "
              f"Cropped {cut} points from each side. Final length = {len(abscissa_shifted)}")
    else:
        print(f"[INFO] Centering: peak is already centered ({ref_max_idx}). No cropping needed.")

    return shifted, abscissa_shifted

def make_material_from_csv(filepath, wl_min, wl_max, verbose = False ):
    """
    Create a Meep material (mp.Medium) using wavelength-dependent n,k data
    from a CSV file with columns: wl [μm], n, k.
    The material is represented using a single LorentzianSusceptibility term
    averaged over the specified wavelength range.
    """
    # --- Load and filter data ---
    df = pd.read_csv(filepath)
    df = df[(df["wl"] >= wl_min) & (df["wl"] <= wl_max)]
    if df.empty:
        raise ValueError("No data in the specified wavelength range.")

    n = df["n"].to_numpy()
    k = df["k"].to_numpy()

    # --- Compute complex dielectric constant ε(λ) = (n + i k)^2 ---
    eps_complex = (n + 1j * k) ** 2
    eps_real = np.mean(eps_complex.real)
    eps_imag = np.mean(eps_complex.imag)

    # --- Convert wavelength range to frequency range (1/μm) ---
    freq_min = 1 / wl_max
    freq_max = 1 / wl_min
    valid_range = mp.FreqRange(min=freq_min, max=freq_max)

    # --- Define LorentzianSusceptibility using averaged response ---
    # This is an approximate single-pole model matching the optical range.
    center_wl = 0.5 * (wl_min + wl_max)
    lorentz_freq = 1 / (center_wl * 1)
    lorentz_gamma = 1 / ((wl_max - wl_min) * 1)
    lorentz_sigma = abs(eps_imag) * 2  # scaling for absorption

    susc = [
        mp.LorentzianSusceptibility(
            frequency=lorentz_freq,
            gamma=lorentz_gamma,
            sigma=lorentz_sigma
        )
    ]

    # --- Define medium ---
    material = mp.Medium(
        epsilon=eps_real,
        E_susceptibilities=susc,
        valid_freq_range=valid_range
    )

    if verbose:
        print(f"Material created from '{filepath}'")
        print(f"  wavelength range: {wl_min:.2f}–{wl_max:.2f} µm")
        print(f"  average ε = {eps_real:.3f} + {eps_imag:.3f}j")
    return material, eps_real

