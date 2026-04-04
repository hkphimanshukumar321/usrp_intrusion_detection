#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: SDR Intrusion Detection - Simulation Demo
# Author: Shubham Gupta, Himanshu Kumar
# Copyright: 2025
# Description: QPSK TX -> AWGN Channel -> Jammer -> RX with spike detection
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import analog
from gnuradio import blocks
import numpy
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import sip
import threading



class intrusion_detection_sim(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "SDR Intrusion Detection - Simulation Demo", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("SDR Intrusion Detection - Simulation Demo")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "intrusion_detection_sim")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 10
        self.samp_rate = samp_rate = 1920000
        self.noise_voltage = noise_voltage = 0.1
        self.jammer_enable = jammer_enable = 0
        self.jammer_amp = jammer_amp = 1.0
        self.excess_bw = excess_bw = 0.35

        ##################################################
        # Blocks
        ##################################################

        self._noise_voltage_range = qtgui.Range(0, 1.0, 0.01, 0.1, 200)
        self._noise_voltage_win = qtgui.RangeWidget(self._noise_voltage_range, self.set_noise_voltage, "AWGN Noise Voltage", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._noise_voltage_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        # Create the options list
        self._jammer_enable_options = [0, 1]
        # Create the labels list
        self._jammer_enable_labels = ['OFF', 'ON']
        # Create the combo box
        # Create the radio buttons
        self._jammer_enable_group_box = Qt.QGroupBox("Jammer" + ": ")
        self._jammer_enable_box = Qt.QVBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._jammer_enable_button_group = variable_chooser_button_group()
        self._jammer_enable_group_box.setLayout(self._jammer_enable_box)
        for i, _label in enumerate(self._jammer_enable_labels):
            radio_button = Qt.QRadioButton(_label)
            self._jammer_enable_box.addWidget(radio_button)
            self._jammer_enable_button_group.addButton(radio_button, i)
        self._jammer_enable_callback = lambda i: Qt.QMetaObject.invokeMethod(self._jammer_enable_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._jammer_enable_options.index(i)))
        self._jammer_enable_callback(self.jammer_enable)
        self._jammer_enable_button_group.buttonClicked[int].connect(
            lambda i: self.set_jammer_enable(self._jammer_enable_options[i]))
        self.top_grid_layout.addWidget(self._jammer_enable_group_box, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._jammer_amp_range = qtgui.Range(0, 5.0, 0.1, 1.0, 200)
        self._jammer_amp_win = qtgui.RangeWidget(self._jammer_amp_range, self.set_jammer_amp, "Jammer Amplitude", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._jammer_amp_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.root_raised_cosine_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                (samp_rate/sps),
                excess_bw,
                (11*sps+1)))
        self.qtgui_time_sink_tx = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "TX Signal", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_tx.set_update_time(0.10)
        self.qtgui_time_sink_tx.set_y_axis(-2, 2)

        self.qtgui_time_sink_tx.set_y_label('Amplitude', "")

        self.qtgui_time_sink_tx.enable_tags(True)
        self.qtgui_time_sink_tx.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_tx.enable_autoscale(True)
        self.qtgui_time_sink_tx.enable_grid(True)
        self.qtgui_time_sink_tx.enable_axis_labels(True)
        self.qtgui_time_sink_tx.enable_control_panel(False)
        self.qtgui_time_sink_tx.enable_stem_plot(False)


        labels = ['I', 'Q', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_tx.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_tx.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_tx.set_line_label(i, labels[i])
            self.qtgui_time_sink_tx.set_line_width(i, widths[i])
            self.qtgui_time_sink_tx.set_line_color(i, colors[i])
            self.qtgui_time_sink_tx.set_line_style(i, styles[i])
            self.qtgui_time_sink_tx.set_line_marker(i, markers[i])
            self.qtgui_time_sink_tx.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_tx_win = sip.wrapinstance(self.qtgui_time_sink_tx.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_tx_win, 2, 0, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_time_sink_spikes = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            "Spike Detection Output", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_spikes.set_update_time(0.10)
        self.qtgui_time_sink_spikes.set_y_axis(-0.5, 1.5)

        self.qtgui_time_sink_spikes.set_y_label('Detection', "")

        self.qtgui_time_sink_spikes.enable_tags(True)
        self.qtgui_time_sink_spikes.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_spikes.enable_autoscale(True)
        self.qtgui_time_sink_spikes.enable_grid(True)
        self.qtgui_time_sink_spikes.enable_axis_labels(True)
        self.qtgui_time_sink_spikes.enable_control_panel(False)
        self.qtgui_time_sink_spikes.enable_stem_plot(True)


        labels = ['Spike Detection', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [2, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['red', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_spikes.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_spikes.set_line_label(i, labels[i])
            self.qtgui_time_sink_spikes.set_line_width(i, widths[i])
            self.qtgui_time_sink_spikes.set_line_color(i, colors[i])
            self.qtgui_time_sink_spikes.set_line_style(i, styles[i])
            self.qtgui_time_sink_spikes.set_line_marker(i, markers[i])
            self.qtgui_time_sink_spikes.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_spikes_win = sip.wrapinstance(self.qtgui_time_sink_spikes.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_spikes_win, 3, 0, 1, 2)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_time_sink_rx = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "RX Signal (After Channel)", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_rx.set_update_time(0.10)
        self.qtgui_time_sink_rx.set_y_axis(-2, 2)

        self.qtgui_time_sink_rx.set_y_label('Amplitude', "")

        self.qtgui_time_sink_rx.enable_tags(True)
        self.qtgui_time_sink_rx.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_rx.enable_autoscale(True)
        self.qtgui_time_sink_rx.enable_grid(True)
        self.qtgui_time_sink_rx.enable_axis_labels(True)
        self.qtgui_time_sink_rx.enable_control_panel(False)
        self.qtgui_time_sink_rx.enable_stem_plot(False)


        labels = ['I', 'Q', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_rx.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_rx.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_rx.set_line_label(i, labels[i])
            self.qtgui_time_sink_rx.set_line_width(i, widths[i])
            self.qtgui_time_sink_rx.set_line_color(i, colors[i])
            self.qtgui_time_sink_rx.set_line_style(i, styles[i])
            self.qtgui_time_sink_rx.set_line_marker(i, markers[i])
            self.qtgui_time_sink_rx.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_rx_win = sip.wrapinstance(self.qtgui_time_sink_rx.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_rx_win, 2, 1, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "RF Spectrum", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-80), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(True)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['Spectrum', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 4, 0, 1, 2)
        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_MUELLER_AND_MULLER,
            sps,
            0.045,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_bpsk().base(),
            digital.IR_MMSE_8TAP,
            128,
            [])
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(0.05, 4, False)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=digital.constellation_qpsk().base(),
            differential=False,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=42,
            block_tags=False)
        self.blocks_uchar_to_float_0 = blocks.uchar_to_float()
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_peak_detector_xb_0 = blocks.peak_detector_fb(1.5, 1.2, 10, 0.001)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_multiply_const_jam = blocks.multiply_const_cc(jammer_enable)
        self.blocks_file_sink_rx = blocks.file_sink(gr.sizeof_gr_complex*1, 'rx_data.dat', False)
        self.blocks_file_sink_rx.set_unbuffered(False)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_sig_source_jammer = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 50000, jammer_amp, 0, 0)
        self.analog_random_source_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 256, 1000))), True)
        self.analog_agc_xx_0 = analog.agc_cc((1e-4), 1.0, 1, 65536)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.analog_agc_xx_0, 0), (self.root_raised_cosine_filter_0, 0))
        self.connect((self.analog_random_source_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.analog_sig_source_jammer, 0), (self.blocks_multiply_const_jam, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.analog_agc_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_file_sink_rx, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_time_sink_rx, 0))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_peak_detector_xb_0, 0))
        self.connect((self.blocks_multiply_const_jam, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_peak_detector_xb_0, 0), (self.blocks_uchar_to_float_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_uchar_to_float_0, 0), (self.qtgui_time_sink_spikes, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.qtgui_time_sink_tx, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.digital_symbol_sync_xx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "intrusion_detection_sim")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.digital_symbol_sync_xx_0.set_sps(self.sps)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.excess_bw, (11*self.sps+1)))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_jammer.set_sampling_freq(self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_rx.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_spikes.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_tx.set_samp_rate(self.samp_rate)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.excess_bw, (11*self.sps+1)))
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)

    def get_noise_voltage(self):
        return self.noise_voltage

    def set_noise_voltage(self, noise_voltage):
        self.noise_voltage = noise_voltage
        self.channels_channel_model_0.set_noise_voltage(self.noise_voltage)

    def get_jammer_enable(self):
        return self.jammer_enable

    def set_jammer_enable(self, jammer_enable):
        self.jammer_enable = jammer_enable
        self._jammer_enable_callback(self.jammer_enable)
        self.blocks_multiply_const_jam.set_k(self.jammer_enable)

    def get_jammer_amp(self):
        return self.jammer_amp

    def set_jammer_amp(self, jammer_amp):
        self.jammer_amp = jammer_amp
        self.analog_sig_source_jammer.set_amplitude(self.jammer_amp)

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.excess_bw, (11*self.sps+1)))




def main(top_block_cls=intrusion_detection_sim, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
