{
  pkgs,
  lib,
  ...
}: {
  packages = with pkgs; [
    # --- Build / dev tools ---
    cmake
    pkg-config
    gcc
    git
    just
    gitleaks
    ruff

    # --- Runtime libs needed for PySide6/Qt at runtime ---
    libGL
    libGLU
    libusb1
    glib
    dbus
    fontconfig
    libxkbcommon
    zlib
    stdenv.cc.cc.lib
    systemd
    freetype
    zstd
    wayland

    # X11 libs (needed even under Wayland for XWayland/fallback apps)
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libXtst
    xorg.libXi
    xorg.libxcb
    xorg.xcbutil
    xorg.xcbutilimage
    xorg.xcbutilkeysyms
    xorg.xcbutilwm
    xorg.xcbutilrenderutil
    xorg.xcbutilcursor
  ];

  languages.python = {
    enable = true;
    package = pkgs.python312;

    venv = {
      enable = true;
      # requirements.txt plus the project itself, installed in editable mode so that
      # `import icorrvision` runs the code in src/. devenv runs pip from the project root.
      requirements = builtins.readFile ./requirements.txt + "-e .\n";
    };
  };

  env = {
    QT_QPA_PLATFORM = "wayland;xcb";

    PKG_CONFIG_PATH = lib.makeSearchPathOutput "dev" "lib/pkgconfig" (with pkgs; [
      zlib
      glib
    ]);
  };

  env.OCL_ICD_VENDORS = "/run/opengl-driver/etc/OpenCL/vendors"; # where NixOS registers the drivers
  # The two settings below were used for the OpenCL route in benchmarks/acceleration_routes.ipynb
  # on the machine the thesis results were produced on (AMD Radeon RX 6700 XT); with current drivers
  # that route also runs without them. They are specific to that hardware: enable them only on a
  # similar AMD RDNA2 GPU if OpenCL finds no device.
  # env.RUSTICL_ENABLE = "radeonsi"; # rusticl exposes no device without it
  # env.HSA_OVERRIDE_GFX_VERSION = "10.3.0"; # ROCm: treat the 6700 XT (gfx1031) as gfx1030

  scripts.dictest.exec = ''
    PYSIDE6_DIR="$(python -c 'import PySide6, os; print(os.path.dirname(PySide6.__file__))')"
    export LD_LIBRARY_PATH="$PYSIDE6_DIR/Qt/lib:${lib.makeLibraryPath (with pkgs; [
      libGL
      libGLU
      freetype
      zstd
      glib
      dbus
      zlib
      stdenv.cc.cc.lib
      fontconfig
      libxkbcommon
      wayland
      xorg.libX11
      xorg.libXext
      xorg.libXrender
      xorg.libXtst
      xorg.libXi
      xorg.libxcb
      xorg.xcbutil
      xorg.xcbutilimage
      xorg.xcbutilkeysyms
      xorg.xcbutilwm
      xorg.xcbutilrenderutil
      xorg.xcbutilcursor
    ])}"
    export QT_PLUGIN_PATH="$PYSIDE6_DIR/Qt/plugins"
    python3 -m icorrvision.gui.main
  '';

  enterShell = ''
        kernel_dir="$DEVENV_STATE/jupyter/kernels/dic-thesis-devenv"
        mkdir -p "$kernel_dir"

        cat > "$kernel_dir/kernel.json" <<KERNELSPEC_EOF
        {
          "argv": ["$(command -v python)", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
          "display_name": "DIC thesis (devenv)",
          "language": "python"
        }
    KERNELSPEC_EOF

        export JUPYTER_PATH="$DEVENV_STATE/jupyter''${JUPYTER_PATH:+:$JUPYTER_PATH}"
        export MOLTEN_KERNEL=dic-thesis-devenv
  '';
}
