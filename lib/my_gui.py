import ipywidgets as widgets
import numpy as np
from IPython.display import display
from configs import conf_test as conf

class LiveInfoPanel:
    def __init__(self, title="Parameters of simulation"):
        self.title = title
        self.style = """
        <style>
            .live-panel { 
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                font-family: Arial, sans-serif;
            }
            .live-panel h3 {
                color: #78d400;
                margin-top: 0;
            }
            .live-panel ul {
                list-style-type: none;
                padding-left: 0;
            }
            .live-panel li {
                margin: 0px 0;
            }
            .live-panel b {
                color: #0078D4;
            }
        </style>
        """

    def _generate_html(self):
        return f"""
        {self.style}
        <div class="live-panel">
            <h3>{self.title}</h3>
            <ul>
                <li>1. Lense </li>
                <li>Aperture: <b>{conf.aperture} μm </b></li>
                <li>Diaphragm: <b>{conf.diaphragm} μm </b></li>
                <li>Orders entire lense : <b>{conf.orders}</b></li>
                <li>Orders of diaphragm : <b>{conf.diaphragm_orders}</b></li>
                <li>Focal length: <b>{conf.f} μm </b></li>
                <li>Numerical aperture: <b>{conf.numerical_aperture} </b></li>
                <li>Height limit of atoms: <b>{conf.at_hei} μm</b> epsilon: <b>{np.around(conf.at_epsilon, decimals = 2)}</b></li>
                <li>Height of substrate: <b>{conf.in_sub} μm</b> epsilon: <b>{np.around(conf.in_epsilon, decimals = 2)}</b></li>
                <li>2. Source </li>
                <li>Broadband from: <b>{conf.inc_start_wavelength} μm</b> to <b>{conf.inc_end_wavelength} μm</b> divided into <b>{conf.inc_nfreq}</b> with center in: <b>{conf.inc_wavelength} μm</b></li>
                <li>Angle of incident: <b>{np.around(conf.angle_list, decimals=1)} °</b></li>               
                <li>3. Simulation </li>
                <li>General resolution is: <b>{conf.min_resolution * conf.coef_resolution} </b></li>
                <li>Grid size is: <b>{conf.dx} μm </b></li>
                <li>Resolution of design zones [x_dim]: <b>{conf.x_dim} pix</b> [h_dim]: <b>{conf.h_dim} pix</b> </li>
                <li>Field of view in focal plane: <b>{conf.spot_ff} μm</b> Resolution: <b>{conf.spot_ff*conf.ff_res} pixels</b>  </li>              
            </ul>
        </div>
        """

    def show(self):
        """Создает новую панель в текущем месте вывода"""
        html = widgets.HTML(self._generate_html())
        display(html)