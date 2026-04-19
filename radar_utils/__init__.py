
# =============================================================================
# radar_utils — Core library for X-Band phased-array radar operations
# =============================================================================
# This package contains all the building blocks for configuring hardware,
# processing radar signals, and running calibration.  Import directly from
# submodules so you can Ctrl-click to see where each function lives:
#
#   from radar_utils.calibration       import enable_stingray_channel, ...
#   from radar_utils.cal_manager       import save_calibration, load_latest_calibration
#   from radar_utils.hardware_setup    import setup_ad9081, setup_stingray, get_radar_data
#   from radar_utils.radar_plotting    import init_rd_gui, plot_fmcw_data
#   from radar_utils.signal_processing import freq_process, apply_cfar_2d, RDRConfig
#   from radar_utils.sync_config       import sys_sync, sync_disable
#   from radar_utils.tracking          import AlphaBetaFilter, MofNConfirmer
#   from radar_utils.tx_rx_cal         import setup, rx_cal_full, tx_cal_full
#   from radar_utils.utils             import maximize_by_title, window_exists
# =============================================================================
