#!/usr/bin/env python3
"""
GUI Application for FTIR UV Curing Analysis
User-friendly graphical interface for the analysis system
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
from pathlib import Path
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from opus_reader import OPUSReader
from ftir_analyzer import FTIRUVCuringAnalyzer
from visualization import FTIRVisualizer
from report_generator import ReportGenerator

class FTIRAnalysisGUI:
    """
    Main GUI application for FTIR UV curing analysis
    """

    def __init__(self, root):
        self.root = root
        self.root.title("FTIR UV Curing Analysis System")
        self.root.geometry("1200x800")

        # Initialize components
        self.opus_reader = OPUSReader()
        self.analyzer = FTIRUVCuringAnalyzer()
        self.visualizer = FTIRVisualizer()
        self.report_generator = ReportGenerator()

        # Data storage
        self.current_data = None
        self.analysis_results = None

        # Threading
        self.analysis_queue = queue.Queue()

        # Create GUI
        self.create_widgets()
        self.create_menu()

        # Start queue monitoring
        self.root.after(100, self.check_queue)

    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load OPUS Files...", command=self.load_opus_files)
        file_menu.add_command(label="Load CSV Data...", command=self.load_csv_data)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results...", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Full Analysis", command=self.run_full_analysis)
        analysis_menu.add_command(label="Quick C=C Analysis", command=self.run_cc_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="PCA Analysis", command=self.run_pca_analysis)
        analysis_menu.add_command(label="Difference Spectra", command=self.calculate_difference_spectra)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)

    def create_widgets(self):
        """Create main GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Data Loading and Parameters
        self.create_data_tab()

        # Tab 2: Analysis Results
        self.create_results_tab()

        # Tab 3: Visualizations
        self.create_visualization_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_data_tab(self):
        """Create data loading and parameter tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data & Parameters")

        # File selection section
        file_section = ttk.LabelFrame(data_frame, text="Data Files", padding=10)
        file_section.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(file_section, text="Load OPUS Files",
                  command=self.load_opus_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_section, text="Load CSV Data",
                  command=self.load_csv_data).pack(side=tk.LEFT, padx=5)

        self.file_info_var = tk.StringVar()
        self.file_info_var.set("No data loaded")
        ttk.Label(file_section, textvariable=self.file_info_var).pack(side=tk.LEFT, padx=20)

        # Analysis parameters section
        params_section = ttk.LabelFrame(data_frame, text="Analysis Parameters", padding=10)
        params_section.pack(fill=tk.X, padx=10, pady=5)

        # Baseline correction
        ttk.Label(params_section, text="Baseline Correction:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.baseline_var = tk.StringVar(value="als")
        baseline_combo = ttk.Combobox(params_section, textvariable=self.baseline_var,
                                     values=["als", "polynomial"], state="readonly")
        baseline_combo.grid(row=0, column=1, padx=5, pady=2)

        # Normalization
        ttk.Label(params_section, text="Normalization:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.norm_var = tk.StringVar(value="max")
        norm_combo = ttk.Combobox(params_section, textvariable=self.norm_var,
                                 values=["max", "area", "snv"], state="readonly")
        norm_combo.grid(row=1, column=1, padx=5, pady=2)

        # Wavenumber range
        ttk.Label(params_section, text="Wavenumber Range:").grid(row=2, column=0, sticky=tk.W, padx=5)
        range_frame = ttk.Frame(params_section)
        range_frame.grid(row=2, column=1, padx=5, pady=2)

        self.wn_min_var = tk.StringVar(value="1000")
        self.wn_max_var = tk.StringVar(value="4000")
        ttk.Entry(range_frame, textvariable=self.wn_min_var, width=8).pack(side=tk.LEFT)
        ttk.Label(range_frame, text=" - ").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.wn_max_var, width=8).pack(side=tk.LEFT)
        ttk.Label(range_frame, text=" cm⁻¹").pack(side=tk.LEFT)

        # Analysis buttons
        button_section = ttk.LabelFrame(data_frame, text="Analysis", padding=10)
        button_section.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_section, text="Run Full Analysis",
                  command=self.run_full_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_section, text="Quick C=C Analysis",
                  command=self.run_cc_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_section, text="PCA Analysis",
                  command=self.run_pca_analysis).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(data_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # Data preview
        preview_section = ttk.LabelFrame(data_frame, text="Data Preview", padding=10)
        preview_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create treeview for data preview
        columns = ('Time', 'Wavenumber_Range', 'Data_Points')
        self.data_tree = ttk.Treeview(preview_section, columns=columns, show='headings', height=8)

        for col in columns:
            self.data_tree.heading(col, text=col.replace('_', ' '))
            self.data_tree.column(col, width=120)

        scrollbar = ttk.Scrollbar(preview_section, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)

        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_results_tab(self):
        """Create analysis results tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Analysis Results")

        # Results text area
        text_frame = ttk.LabelFrame(results_frame, text="Analysis Summary", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        results_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Export buttons
        export_frame = ttk.LabelFrame(results_frame, text="Export Results", padding=10)
        export_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(export_frame, text="Save Text Report",
                  command=self.save_text_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Save HTML Report",
                  command=self.save_html_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Save Excel Data",
                  command=self.save_excel_report).pack(side=tk.LEFT, padx=5)

    def create_visualization_tab(self):
        """Create visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualizations")

        # Plot selection
        plot_frame = ttk.LabelFrame(viz_frame, text="Plot Selection", padding=10)
        plot_frame.pack(fill=tk.X, padx=10, pady=5)

        self.plot_type_var = tk.StringVar(value="spectral_evolution")
        plot_types = [
            ("Spectral Evolution", "spectral_evolution"),
            ("Kinetic Curves", "kinetic_curves"),
            ("Difference Spectra", "difference_spectra"),
            ("PCA Analysis", "pca_analysis"),
            ("Summary Plot", "summary")
        ]

        for i, (text, value) in enumerate(plot_types):
            ttk.Radiobutton(plot_frame, text=text, variable=self.plot_type_var,
                           value=value, command=self.update_plot).grid(row=0, column=i, padx=5)

        # Matplotlib canvas
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Plot controls
        controls_frame = ttk.LabelFrame(viz_frame, text="Plot Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(controls_frame, text="Update Plot",
                  command=self.update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Save Plot",
                  command=self.save_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Clear Plot",
                  command=self.clear_plot).pack(side=tk.LEFT, padx=5)

    def load_opus_files(self):
        """Load OPUS files from directory"""
        directory = filedialog.askdirectory(title="Select Directory with OPUS Files")
        if not directory:
            return

        self.status_var.set("Loading OPUS files...")
        self.progress_var.set(10)

        try:
            # Find OPUS files
            opus_files = self.opus_reader.find_opus_files(directory)

            if not opus_files:
                messagebox.showwarning("No Files", "No OPUS files found in the selected directory")
                self.status_var.set("Ready")
                self.progress_var.set(0)
                return

            self.progress_var.set(30)

            # Convert to CSV
            csv_output_dir = Path(directory) / 'csv_output'
            conversion_results = self.opus_reader.batch_convert(directory, str(csv_output_dir))

            self.progress_var.set(60)

            # Create integrated dataset
            if conversion_results['csv_files']:
                integrated_file = str(Path(directory) / 'integrated_spectra.csv')
                self.opus_reader.create_integrated_dataset(conversion_results['csv_files'], integrated_file)

                # Load the integrated data
                success = self.analyzer.load_data(integrated_file)
                if success:
                    self.current_data = integrated_file
                    self.update_data_info()
                    self.update_data_preview()
                    self.status_var.set(f"Loaded {len(conversion_results['successful'])} OPUS files")
                else:
                    messagebox.showerror("Error", "Failed to load integrated dataset")
            else:
                messagebox.showerror("Error", "No OPUS files were successfully converted")

            self.progress_var.set(100)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load OPUS files: {str(e)}")
            self.status_var.set("Error loading files")

        finally:
            self.progress_var.set(0)

    def load_csv_data(self):
        """Load CSV data file"""
        filename = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        self.status_var.set("Loading CSV data...")

        try:
            success = self.analyzer.load_data(filename)
            if success:
                self.current_data = filename
                self.update_data_info()
                self.update_data_preview()
                self.status_var.set("CSV data loaded successfully")
            else:
                messagebox.showerror("Error", "Failed to load CSV data")
                self.status_var.set("Error loading data")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV data: {str(e)}")
            self.status_var.set("Error loading data")

    def update_data_info(self):
        """Update data information display"""
        if self.analyzer.data is not None:
            info = f"Time points: {len(self.analyzer.exposure_times)}, "
            info += f"Wavenumbers: {len(self.analyzer.wavenumbers)}, "
            info += f"Time range: {min(self.analyzer.exposure_times):.1f}-{max(self.analyzer.exposure_times):.1f}s"
            self.file_info_var.set(info)
        else:
            self.file_info_var.set("No data loaded")

    def update_data_preview(self):
        """Update data preview table"""
        # Clear existing items
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        if self.analyzer.data is not None:
            # Group data by exposure time
            for time in sorted(self.analyzer.exposure_times):
                time_data = self.analyzer.data[self.analyzer.data['ExposureTime'] == time]
                wn_min = time_data['Wavenumber'].min()
                wn_max = time_data['Wavenumber'].max()
                n_points = len(time_data)

                self.data_tree.insert('', 'end', values=(
                    f"{time:.1f}s",
                    f"{wn_min:.0f}-{wn_max:.0f}",
                    str(n_points)
                ))

    def run_full_analysis(self):
        """Run complete analysis in background thread"""
        if self.analyzer.data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        # Start analysis in background thread
        self.status_var.set("Running full analysis...")
        self.progress_var.set(0)

        analysis_thread = threading.Thread(target=self._run_full_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()

    def _run_full_analysis_thread(self):
        """Background thread for full analysis"""
        try:
            # Update progress
            self.analysis_queue.put(("progress", 20))

            # Run automated analysis
            results = self.analyzer.run_automated_analysis(
                baseline_method=self.baseline_var.get(),
                norm_method=self.norm_var.get()
            )

            self.analysis_queue.put(("progress", 80))

            # Store results
            self.analysis_results = results

            self.analysis_queue.put(("progress", 100))
            self.analysis_queue.put(("complete", "Full analysis completed successfully"))

        except Exception as e:
            self.analysis_queue.put(("error", str(e)))

    def run_cc_analysis(self):
        """Run quick C=C analysis"""
        if self.analyzer.data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        self.status_var.set("Running C=C analysis...")

        try:
            # Preprocess data
            self.analyzer.preprocess_data(
                baseline_method=self.baseline_var.get(),
                norm_method=self.norm_var.get()
            )

            # Analyze C=C region
            cc_results = self.analyzer.analyze_cc_consumption((1620, 1640))

            # Display results
            self.display_cc_results(cc_results)
            self.status_var.set("C=C analysis completed")

        except Exception as e:
            messagebox.showerror("Error", f"C=C analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")

    def run_pca_analysis(self):
        """Run PCA analysis"""
        if self.analyzer.data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        self.status_var.set("Running PCA analysis...")

        try:
            # Preprocess data if not already done
            if self.analyzer.processed_data is None:
                self.analyzer.preprocess_data(
                    baseline_method=self.baseline_var.get(),
                    norm_method=self.norm_var.get()
                )

            # Run PCA
            pca_results = self.analyzer.perform_pca_analysis()

            # Display results
            self.display_pca_results(pca_results)
            self.status_var.set("PCA analysis completed")

        except Exception as e:
            messagebox.showerror("Error", f"PCA analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")

    def display_cc_results(self, cc_results):
        """Display C=C analysis results"""
        self.results_text.delete(1.0, tk.END)

        text = "C=C DOUBLE BOND CONSUMPTION ANALYSIS\n"
        text += "="*50 + "\n\n"

        text += f"Wavenumber Range: {cc_results['wavenumber_range'][0]:.0f}-{cc_results['wavenumber_range'][1]:.0f} cm⁻¹\n"
        text += f"Final Conversion: {max(cc_results['conversion_percent']):.2f}%\n\n"

        text += "KINETIC MODELS:\n"
        text += "-"*30 + "\n"

        if 'kinetic_models' in cc_results:
            for model_name, model_data in cc_results['kinetic_models'].items():
                if 'error' not in model_data:
                    text += f"\n{model_name.upper()}:\n"
                    text += f"  R² Value: {model_data.get('r_squared', 0):.4f}\n"
                    text += f"  Rate Constant: {model_data.get('rate_constant', 0):.2e} s⁻¹\n"
                    if 'c_max' in model_data:
                        text += f"  Maximum Conversion: {model_data['c_max']:.2f}%\n"

        self.results_text.insert(1.0, text)

    def display_pca_results(self, pca_results):
        """Display PCA analysis results"""
        self.results_text.delete(1.0, tk.END)

        text = "PRINCIPAL COMPONENT ANALYSIS\n"
        text += "="*50 + "\n\n"

        text += "EXPLAINED VARIANCE:\n"
        text += "-"*30 + "\n"

        for i in range(min(5, len(pca_results['explained_variance']))):
            var_pct = pca_results['explained_variance'][i] * 100
            cum_var_pct = pca_results['cumulative_variance'][i] * 100
            text += f"PC{i+1}: {var_pct:.1f}% (Cumulative: {cum_var_pct:.1f}%)\n"

        text += f"\nFirst 3 PCs explain {pca_results['cumulative_variance'][2]*100:.1f}% of total variance\n"

        self.results_text.insert(1.0, text)

    def update_plot(self):
        """Update the current plot"""
        if self.analyzer.data is None:
            return

        self.fig.clear()

        try:
            plot_type = self.plot_type_var.get()

            if plot_type == "spectral_evolution":
                if self.analyzer.processed_data is not None:
                    ax = self.fig.add_subplot(111)

                    # Plot spectral evolution
                    exposure_times = sorted(self.analyzer.processed_data['ExposureTime'].unique())[:5]
                    colors = plt.cm.viridis(np.linspace(0, 1, len(exposure_times)))

                    for i, time in enumerate(exposure_times):
                        time_data = self.analyzer.processed_data[
                            self.analyzer.processed_data['ExposureTime'] == time
                        ].sort_values('Wavenumber')

                        ax.plot(time_data['Wavenumber'], time_data['ProcessedAbsorbance'],
                               color=colors[i], label=f'{time}s', alpha=0.8)

                    ax.set_xlabel('Wavenumber (cm⁻¹)')
                    ax.set_ylabel('Normalized Absorbance')
                    ax.set_title('FTIR Spectral Evolution')
                    ax.legend()
                    ax.invert_xaxis()
                    ax.grid(True, alpha=0.3)

            elif plot_type == "kinetic_curves" and self.analysis_results:
                if 'region_analysis' in self.analysis_results:
                    region_data = self.analysis_results['region_analysis']

                    # Plot first region's kinetics
                    first_region = list(region_data.keys())[0]
                    results = region_data[first_region]

                    ax = self.fig.add_subplot(111)
                    times = results['exposure_times']
                    conversion = results['conversion_percent']

                    ax.scatter(times, conversion, color='red', s=50, alpha=0.7, label='Experimental')

                    ax.set_xlabel('Exposure Time (s)')
                    ax.set_ylabel('Conversion (%)')
                    ax.set_title(f'Kinetic Curve - {first_region.replace("_", " ").title()}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            elif plot_type == "pca_analysis" and self.analysis_results:
                if 'pca_analysis' in self.analysis_results:
                    pca_data = self.analysis_results['pca_analysis']

                    ax = self.fig.add_subplot(111)
                    scores = pca_data['scores']
                    times = pca_data['exposure_times']

                    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=times, cmap='viridis', s=100)
                    ax.set_xlabel(f'PC1 ({pca_data["explained_variance"][0]*100:.1f}%)')
                    ax.set_ylabel(f'PC2 ({pca_data["explained_variance"][1]*100:.1f}%)')
                    ax.set_title('PCA Scores Plot')
                    ax.grid(True, alpha=0.3)

                    # Add colorbar
                    cbar = self.fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Exposure Time (s)')

            self.canvas.draw()

        except Exception as e:
            print(f"Plot update failed: {str(e)}")

    def save_plot(self):
        """Save current plot"""
        filename = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Plot saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def clear_plot(self):
        """Clear the current plot"""
        self.fig.clear()
        self.canvas.draw()

    def save_text_report(self):
        """Save text report"""
        if not self.analysis_results:
            messagebox.showwarning("No Results", "Please run analysis first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Text Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.report_generator.generate_text_report(self.analysis_results, filename)
                self.status_var.set(f"Text report saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")

    def save_html_report(self):
        """Save HTML report"""
        if not self.analysis_results:
            messagebox.showwarning("No Results", "Please run analysis first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save HTML Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.report_generator.generate_html_report(self.analysis_results, filename)
                self.status_var.set(f"HTML report saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")

    def save_excel_report(self):
        """Save Excel report"""
        if not self.analysis_results:
            messagebox.showwarning("No Results", "Please run analysis first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Excel Report",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.report_generator.generate_excel_report(self.analysis_results, filename)
                self.status_var.set(f"Excel report saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")

    def save_results(self):
        """Save all results"""
        if not self.analysis_results:
            messagebox.showwarning("No Results", "Please run analysis first")
            return

        directory = filedialog.askdirectory(title="Select Directory to Save Results")
        if not directory:
            return

        try:
            output_dir = Path(directory)

            # Save reports
            self.report_generator.generate_text_report(
                self.analysis_results,
                str(output_dir / 'analysis_report.txt')
            )
            self.report_generator.generate_html_report(
                self.analysis_results,
                str(output_dir / 'analysis_report.html')
            )
            self.report_generator.generate_excel_report(
                self.analysis_results,
                str(output_dir / 'analysis_data.xlsx')
            )

            # Save plots
            if self.analysis_results.get('processed_data') is not None:
                fig1 = self.visualizer.plot_spectral_evolution(
                    self.analysis_results['processed_data'],
                    save_path=str(output_dir / 'spectral_evolution.png')
                )
                plt.close(fig1)

            if 'region_analysis' in self.analysis_results:
                fig2 = self.visualizer.plot_kinetic_curves(
                    self.analysis_results['region_analysis'],
                    save_path=str(output_dir / 'kinetic_curves.png')
                )
                plt.close(fig2)

            if 'pca_analysis' in self.analysis_results:
                fig3 = self.visualizer.plot_pca_analysis(
                    self.analysis_results['pca_analysis'],
                    save_path=str(output_dir / 'pca_analysis.png')
                )
                plt.close(fig3)

            self.status_var.set(f"All results saved to: {directory}")
            messagebox.showinfo("Success", f"All results saved to:\n{directory}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def calculate_difference_spectra(self):
        """Calculate and display difference spectra"""
        if self.analyzer.processed_data is None:
            messagebox.showwarning("No Data", "Please preprocess data first")
            return

        try:
            diff_spectra = self.analyzer.calculate_difference_spectra()

            if not diff_spectra.empty:
                # Update plot to show difference spectra
                self.plot_type_var.set("difference_spectra")

                self.fig.clear()
                ax = self.fig.add_subplot(111)

                exposure_times = sorted(diff_spectra['ExposureTime'].unique())
                colors = plt.cm.viridis(np.linspace(0, 1, len(exposure_times)))

                for i, time in enumerate(exposure_times):
                    time_data = diff_spectra[diff_spectra['ExposureTime'] == time].sort_values('Wavenumber')
                    ax.plot(time_data['Wavenumber'], time_data['DifferenceAbsorbance'],
                           color=colors[i], label=f'{time}s', alpha=0.8)

                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_xlabel('Wavenumber (cm⁻¹)')
                ax.set_ylabel('Absorbance Difference')
                ax.set_title('Difference Spectra')
                ax.legend()
                ax.invert_xaxis()
                ax.grid(True, alpha=0.3)

                self.canvas.draw()
                self.status_var.set("Difference spectra calculated")
            else:
                messagebox.showinfo("Info", "No difference spectra data available")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate difference spectra: {str(e)}")

    def check_queue(self):
        """Check for messages from background threads"""
        try:
            while True:
                message_type, message_data = self.analysis_queue.get_nowait()

                if message_type == "progress":
                    self.progress_var.set(message_data)
                elif message_type == "complete":
                    self.status_var.set(message_data)
                    self.progress_var.set(0)
                    self.display_full_results()
                elif message_type == "error":
                    messagebox.showerror("Analysis Error", f"Analysis failed: {message_data}")
                    self.status_var.set("Analysis failed")
                    self.progress_var.set(0)

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.check_queue)

    def display_full_results(self):
        """Display full analysis results"""
        if not self.analysis_results:
            return

        self.results_text.delete(1.0, tk.END)

        text = "COMPREHENSIVE FTIR UV CURING ANALYSIS RESULTS\n"
        text += "="*60 + "\n\n"

        # Analysis parameters
        if 'metadata' in self.analysis_results:
            metadata = self.analysis_results['metadata']
            text += "ANALYSIS PARAMETERS:\n"
            text += "-"*30 + "\n"
            text += f"Baseline Method: {metadata.get('baseline_method', 'Unknown')}\n"
            text += f"Normalization: {metadata.get('normalization_method', 'Unknown')}\n"
            text += f"Time Points: {len(metadata.get('exposure_times', []))}\n"
            text += f"Wavenumber Range: {metadata.get('wavenumber_range', 'Unknown')} cm⁻¹\n\n"

        # Region analysis
        if 'region_analysis' in self.analysis_results:
            text += "CHEMICAL REGION ANALYSIS:\n"
            text += "-"*30 + "\n"

            for region_name, results in self.analysis_results['region_analysis'].items():
                final_conv = max(results['conversion_percent'])
                text += f"\n{region_name.upper().replace('_', ' ')}:\n"
                text += f"  Final Conversion: {final_conv:.2f}%\n"

                if 'kinetic_models' in results:
                    best_model = None
                    best_r2 = -1
                    for model_name, model_data in results['kinetic_models'].items():
                        if 'r_squared' in model_data and model_data['r_squared'] > best_r2:
                            best_r2 = model_data['r_squared']
                            best_model = model_name

                    if best_model:
                        text += f"  Best Model: {best_model} (R² = {best_r2:.4f})\n"

        # PCA results
        if 'pca_analysis' in self.analysis_results:
            pca_data = self.analysis_results['pca_analysis']
            text += "\nPRINCIPAL COMPONENT ANALYSIS:\n"
            text += "-"*30 + "\n"
            text += f"PC1: {pca_data['explained_variance'][0]*100:.1f}% variance\n"
            text += f"PC2: {pca_data['explained_variance'][1]*100:.1f}% variance\n"
            text += f"PC3: {pca_data['explained_variance'][2]*100:.1f}% variance\n"
            text += f"Cumulative (3 PCs): {pca_data['cumulative_variance'][2]*100:.1f}%\n"

        self.results_text.insert(1.0, text)

        # Update plot
        self.update_plot()

    def show_help(self):
        """Show help dialog"""
        help_text = """
FTIR UV Curing Analysis System - User Guide

1. LOADING DATA:
   - Use 'Load OPUS Files' to process Bruker OPUS files
   - Use 'Load CSV Data' to load pre-processed CSV data
   - Required CSV format: Wavenumber, Absorbance, ExposureTime, Filename

2. ANALYSIS PARAMETERS:
   - Baseline Correction: ALS (recommended) or Polynomial
   - Normalization: Max (recommended), Area, or SNV
   - Wavenumber Range: Specify analysis range in cm⁻¹

3. ANALYSIS OPTIONS:
   - Full Analysis: Complete automated analysis
   - Quick C=C Analysis: Focus on C=C consumption (1620-1640 cm⁻¹)
   - PCA Analysis: Principal component analysis

4. VISUALIZATIONS:
   - Spectral Evolution: Time-resolved spectra
   - Kinetic Curves: Conversion vs time
   - Difference Spectra: Spectral changes
   - PCA Analysis: Principal component plots

5. EXPORTING RESULTS:
   - Text Report: Summary in text format
   - HTML Report: Interactive web report
   - Excel Data: Detailed data tables
   - Save All: Complete results package

For more information, see the documentation files.
        """

        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("600x500")

        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(help_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)

    def show_about(self):
        """Show about dialog"""
        about_text = """
FTIR UV Curing Analysis System
Version 1.0.0

A comprehensive Python application for automated analysis of
Fourier Transform Infrared (FTIR) spectroscopy data from
UV curing processes.

Features:
• Bruker OPUS file reading and conversion
• Automated spectral preprocessing
• UV curing kinetic analysis
• Principal component analysis
• Comprehensive reporting

Developed for research and educational purposes.

© 2025 FTIR Analysis System
        """

        messagebox.showinfo("About", about_text)


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = FTIRAnalysisGUI(root)

    # Set window icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()


if __name__ == "__main__":
    main()