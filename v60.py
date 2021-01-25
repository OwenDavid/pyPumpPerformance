#v60.py pump performance model
#copyright Owen David 2021


import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

matplotlib.use('WXAgg',warn=False, force=True)

#inital pump parameters

disp_pump = 130.0 ##cm3/rev
n_input = 2100 #rpm
n_torquelimiter_set = 2100 #rpm - set on dyno
prs_pump = 360 #bar
torque_limit = 455 #nm
prs_relief = 380 #bar
prs_comp_hyst = 4.0 #bar
prs_deltap = 27 #bar


#constant pump parameters

pump_leakage_coefficient = -0.04410281962 #(l/min)/bar
pump_mech_eff = 0.95

# results from inline hydraulik testing


#derived constants

prs_max =min(prs_pump,prs_relief)



#setup figure and axes
fig, ax_pq = plt.subplots(1, figsize=(12,6), dpi=100, facecolor='#edf1f2')
plt.subplots_adjust(left=0.25, bottom=0.3)
ax_pq.set_facecolor('#f2f2f2')


#set axis bounds
ax_pq.set_ylim(0,440)
ax_pq.set_xlim(0,420)


#set axis labels and plot title
ax_pq.set_xlabel('P - pressure (bar)')
ax_pq.set_ylabel('Q - flow rate (l/min)')
ax_pq.set_title('Pump Performance Chart', color='grey')


#set grid and layout
ax_pq.grid(linestyle='--',linewidth=0.3,alpha=0.3)


#setup input widgets
#sliders
axrpm= plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='#f2f2f2')
axcomp= plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='#f2f2f2')
axnm= plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='#f2f2f2')
#radio buttons
axpumpdisp_set = fig.add_axes([0.025, 0.06, 0.05, 0.12], facecolor='#f2f2f2')
#text box
axnm_set = plt.axes([0.25, 0.005, 0.04, 0.04])
axdeltap_set =plt.axes([0.45, 0.005, 0.03, 0.04])
#buttons
ax_reset = plt.axes([0.025, 0.01, 0.05, 0.04])


button_reset = Button(ax_reset, 'Reset', color='#f2f2f2', hovercolor='0.975')
radios_pumpdisp = RadioButtons(axpumpdisp_set, ('60','110', '130', '190'), active=2,activecolor='steelblue')
text_nmset= TextBox(axnm_set, 'Torque Limiter (rpm)   ', initial=str(n_torquelimiter_set))
text_deltapset = TextBox(axdeltap_set, 'Standby Pressure (bar)   ', initial=str(prs_deltap))
slider_rpm = Slider(axrpm, 'Shaft Speed (rpm)', 0, 2500, valinit=n_input, valstep=2, color='steelblue',alpha=0.5)
slider_comp = Slider(axcomp, 'Pump Comp (bar)', 0, 400, valinit=prs_pump, valstep=1,color='steelblue',alpha=0.3)
slider_nm = Slider(axnm, 'Torque Limiter (Nm)', 5, 1000, valinit=torque_limit, valstep=5,color='steelblue',alpha=0.15)


#set data field labels
shaft_pow = fig.text(0.025,0.75,'Shaft Power Limit:',color='darkgrey')
hydro_pow = fig.text(0.025,0.85,'Hydro Corner Power:',color='darkgrey')
max_flow = fig.text(0.025,0.6,'Max Flow:',color='darkgrey')
min_flow = fig.text(0.025,0.5,'Min Flow:',color='darkgrey')
comp_bar = fig.text(0.025,0.35,'Torque Limiting range:',color='darkgrey')


#update pump performance after parameter change
def update(val):
    
    #get new data from widgets
    if slider_comp.val > 400:
        new_comp = 400
    else:
        new_comp = slider_comp.val
    
    
    new_rpm = slider_rpm.val
    new_nm = slider_nm.val
    new_disp = int(radios_pumpdisp.value_selected)
    new_torquelimiter_set = float(text_nmset.text)
    new_delta =  float(text_deltapset.text)


    #set bounding curves and cornerpower marker (grey dashed lines)
    line_flowtheoretical.set_ydata(q_theoretical(new_rpm,new_disp)) #max theoretical flow
    line_p_comp_max.set_xdata(new_comp) #max theoretical pump pressure
    marker_cornerpower[0].set_data(new_comp,q_theoretical(new_rpm,new_disp)) #max theoretical corner power
    
    
    #set pump outline curves (blue lines)
    temp_prscomp,temp_flowcomp,prs_volcomp = flow_cp(new_comp,new_rpm,new_disp,new_comp,prs_comp_hyst,pump_leakage_coefficient)
    temp_prsvol,temp_flowvol = flow_act(0,prs_volcomp,new_rpm,new_disp,pump_leakage_coefficient)
    p_comp_slope, p_comp_intercept = flow_cp(new_comp,new_rpm,new_disp,new_comp,prs_comp_hyst,pump_leakage_coefficient, True)
    
    line_flowcomp[0].set_data(temp_prscomp,temp_flowcomp)
    line_flowactual[0].set_data(temp_prsvol,temp_flowvol)


    #calculate power/flow and update metrics
    hydro_power = temp_prsvol[-1] * temp_flowvol[-1] /600.
    shaft_power = new_nm * new_torquelimiter_set / 9550.0
    
    hydro_kW.set_text('%5.1f'%hydro_power+' Kw')
    shaft_kW.set_text('%5.1f'%shaft_power+' Kw')
    
    lmin_max = q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient)
    max_lmin.set_text('%5.1f'%(lmin_max)+' l/min')
    
    temp_prsfunc,temp_flowfunc,_ = flow_cp(new_comp-new_delta,new_rpm,new_disp,new_comp-new_delta,prs_comp_hyst,pump_leakage_coefficient)
    
    line_flowfunction[0].set_data(temp_prsfunc,temp_flowfunc)
    
    
    if shaft_power < hydro_power:
        
        prs_start_limiter = start_limiter(new_rpm,new_torquelimiter_set,new_disp,new_nm,pump_leakage_coefficient)
        prs_stop_limiter = stop_limiter(p_comp_slope, p_comp_intercept, new_nm)
        temp_prstorque,temp_flowtorque = flow_torque_limiter(prs_start_limiter,prs_stop_limiter,new_torquelimiter_set,new_nm)
        
        line_flowtorquelimiter[0].set_data(temp_prstorque,temp_flowtorque) 
        
        
        text_minLmin.set_text('%5.1f'%(line_flowtorquelimiter[0].get_ydata()[-1])+' l/min')
        text_minPct.set_text('%4.1f'% ( line_flowtorquelimiter[0].get_ydata()[-1] / line_flowactual[0].get_ydata()[0] * 100. ) +' % of max')
        text_minLmin.set_color('orange')
        comp_start.set_text('%4.1f'% (prs_start_limiter-new_delta) +' - '+ '%4.1f'% (prs_stop_limiter-new_delta)+ ' bar')
        comp_start.set_color('orange')
        
        
        fill_x=[(line_flowactual[0].get_xdata()[-1])]
        fill_x.extend(temp_prstorque)
        fill_x.append( (line_flowactual[0].get_xdata()[-1]) )

        fill_y=[(line_flowactual[0].get_ydata()[-1])]
        fill_y.extend(temp_flowtorque)
        fill_y.append( (line_flowactual[0].get_ydata()[-1]))

        fill_torquelimiter[0].set_xy(np.array([fill_x,fill_y]).T)
        fill_torquelimiter[0].set_visible(True)
        
        
        pressure_function = [new_delta,new_delta]
        pressure_function.extend([p-new_delta for p in temp_prstorque])
        pressure_function.append(new_comp-new_delta)
        flow_functions = [f for f in temp_flowtorque]
        flow_functions.append(0.0)
        flow_functions.insert(0,q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient))
        flow_functions.insert(0,0)
        
        line_flowtorquelimiterfunction[0].set_data(pressure_function,flow_functions)
        
        
        deltap_noflowmax_x = [val for val in pressure_function[2:-1]]
        deltap_noflowmax_x.extend([new_comp-new_delta])
        deltap_noflowmax_x.extend([new_comp])
        new_pressure = [p for p in temp_prstorque]
        new_pressure.reverse()
        deltap_noflowmax_x.extend(new_pressure)
        deltap_noflowmax_x.append(pressure_function[2])
        
        deltap_noflowmax_y = [val for val in flow_functions[2:-1]]
        deltap_noflowmax_y.extend([0,0])
        fflow = [val for val in flow_functions[2:-1]]
        fflow.reverse()
        
        deltap_noflowmax_y.extend(fflow)
        deltap_noflowmax_y.append(flow_functions[2])
    
        deltap_noflowmax[0].set_xy(np.array([deltap_noflowmax_x,deltap_noflowmax_y]).T)
        deltap_noflowmax[0].set_visible(True)
       
       
        flow_zone_x = [new_delta, prs_start_limiter-new_delta, prs_start_limiter-new_delta,new_delta,new_delta]    
        flow_zone_y = [0,0,fflow[-2],q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient),0]
       
        flow_zone_fill[0].set_xy(np.array([flow_zone_x,flow_zone_y]).T)
       
        
        limit_zone_x = [prs_start_limiter-new_delta,new_comp-new_delta]
        new_pressure_function = [p for p in pressure_function[2:-1]]
        new_pressure_function.reverse()
        limit_zone_x.extend(new_pressure_function)
        limit_zone_x.append(prs_start_limiter-new_delta)
        
        limit_zone_y = [0,0]
        limit_zone_y.extend(fflow)
        limit_zone_y.append(0)
        limit_zone_fill[0].set_xy(np.array([limit_zone_x,limit_zone_y,]).T)
        limit_zone_fill[0].set_visible(True)
        
        
    else:
        
        line_flowtorquelimiter[0].set_data([],[])
        line_flowtorquelimiterfunction[0].set_data([],[])
        fill_torquelimiter[0].set_visible(False)
        
        text_minLmin.set_text('%5.1f'%(line_flowactual[0].get_ydata()[-1])+' l/min')
        text_minPct.set_text('%4.1f'% ( line_flowactual[0].get_ydata()[-1] / line_flowactual[0].get_ydata()[0] * 100. ) +' % of max')
        text_minLmin.set_color('green')
        comp_start.set_text('Torque Limiter not active')
        comp_start.set_color('green')
        
        temp_prsfunc,temp_flowfunc,_ = flow_cp(new_comp-new_delta,new_rpm,new_disp,new_comp-new_delta,prs_comp_hyst,pump_leakage_coefficient)
        
        line_flowfunction[0].set_data(temp_prsfunc,temp_flowfunc)
        
        
        deltap_noflowmax_x = [new_comp-new_delta,new_comp,line_flowactual[0].get_xdata()[-1] ,line_flowfunction[0].get_xdata()[-1] ]
        
        deltap_noflowmax_y = [0,0,line_flowactual[0].get_ydata()[-1],line_flowfunction[0].get_ydata()[-1]]
        
        deltap_noflowmax[0].set_xy(np.array([deltap_noflowmax_x,deltap_noflowmax_y]).T)
        deltap_noflowmax[0].set_visible(True)        
        
        flow_zone_x = [new_delta, new_comp-new_delta, line_flowfunction[0].get_xdata()[-1],new_delta,new_delta]
        
        flow_zone_y = [0,0,line_flowfunction[0].get_ydata()[-1],q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient),0]
        
        flow_zone_fill[0].set_xy(np.array([flow_zone_x,flow_zone_y]).T)
        
        line_flowtorquelimiterfunction[0].set_data(flow_zone_x[1:],flow_zone_y[1:])
        
        
        limit_zone_fill[0].set_visible(False)
        

        
        
    temp_prstorque,temp_flowtorque = flow_torque_limiter(1,400,new_torquelimiter_set,new_nm)
    line_torquelimiter_ref[0].set_data(temp_prstorque,temp_flowtorque)

    deltap_noflow_x = [0,new_delta,new_delta,0]
    deltap_noflow_y = [0,0,q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient),q_theoretical(new_rpm,new_disp)]
    
    deltap_noflow_fill[0].set_xy(np.array([deltap_noflow_x,deltap_noflow_y]).T)
    
    temp_prsfunc,temp_flowfunc,_ = flow_cp(new_comp-new_delta,new_rpm,new_disp,new_comp-new_delta,prs_comp_hyst,pump_leakage_coefficient)
    line_flowfunction[0].set_data(temp_prsfunc,temp_flowfunc)
    
    
    fig.canvas.draw_idle()



def button_reset_on_clicked(mouse_event):
    slider_rpm.reset()
    slider_comp.reset()
    slider_nm.reset()
    radios_pumpdisp.set_active(2)


def radios_pumpdisp_on_clicked(label):
    max_flow_pump(label)
    
    if radios_pumpdisp.value_selected == '190':
        slider_nm.set_val(895)
        slider_rpm.set_val(1900)
        slider_comp.set_val(360)
    elif radios_pumpdisp.value_selected == '130':
        slider_nm.set_val(455)
        slider_rpm.set_val(2100)
        slider_comp.set_val(360)
    else:
        update(label)


def onclick(event):
        
    if event.inaxes in [ax_pq]:
    
        click_prs = round(event.xdata)
        click_flow = round(event.ydata)  
    
        if event.button == 3 and click_prs <=400:
            h_power = click_prs * click_flow / 600.
            new_torque = h_power / float(text_nmset.text) * 9550.0
            slider_nm.set_val(round(new_torque))
        
        elif event.button == 1 and click_prs <=400:
            slider_comp.set_val(click_prs)
            new_disp = int(radios_pumpdisp.value_selected)
            new_rpm = click_flow * 1000.0 / new_disp
            if new_rpm <=2500:
                slider_rpm.set_val(round(new_rpm))
            else:
                slider_rpm.set_val(2500)

    
class Cursor:
    """
    A cross hair cursor.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='steelblue', lw=0.8, ls='--',alpha=0.15)
        self.vertical_line = ax.axvline(color='steelblue', lw=0.8, ls='--',alpha=0.15)
        # text location in axes coordinates
        #self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        #self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            #self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()



#set actions for input widgets
text_nmset.on_submit(update)
text_deltapset.on_submit(update)
slider_rpm.on_changed(update)
slider_comp.on_changed(update)
slider_nm.on_changed(update)
button_reset.on_clicked(button_reset_on_clicked)
radios_pumpdisp.on_clicked(radios_pumpdisp_on_clicked)

cursor = Cursor(ax_pq)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid_1 = fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)



#pump peformance functions

def q_theoretical(shaft_speed,pump_disp):
    return (shaft_speed * pump_disp / 1000.0)


def q_actual(shaft_speed,pump_disp,prs,leakage_coeff):
    q = q_theoretical(shaft_speed,pump_disp) 
    return q + (prs * leakage_coeff)


def flow_act(p0,p1,shaft_speed,pump_disp,leakage_coeff):
    q0 = q_actual(shaft_speed,pump_disp,p0,leakage_coeff)
    q1 = q_actual(shaft_speed,pump_disp,p1,leakage_coeff)
    
    return (p0,p1),(q0,q1)
    
    
def flow_cp(p_0,shaft_speed,pump_disp,prs_max,prs_comp_hyst,coeff_leakage,coeffs=False):
    #pump compensator characteristic 

    coeff_leakage_intercept = q_theoretical(shaft_speed,pump_disp) 
    p1= prs_max - prs_comp_hyst
    q1= q_actual(shaft_speed,pump_disp,p1,coeff_leakage)
    p2= prs_max
    q2= 0 

    x = np.array((p1,p2)).reshape((-1, 1))
    y = np.array((q1,q2))
    model = LinearRegression().fit(x, y)

    p_comp_slope = model.coef_[0]
    p_comp_intercept = model.intercept_

    #corner power pressure of pump outline curves (blue lines)
    prs_volcomp = ( p_comp_intercept - coeff_leakage_intercept ) / (coeff_leakage - p_comp_slope)

    q_0 = p_comp_slope*p_0 + p_comp_intercept
    p_1 = p_0 - prs_comp_hyst
    q_1 = p_comp_slope*p_1 + p_comp_intercept
    
    if coeffs == True:
        return p_comp_slope, p_comp_intercept
    else:
        return (p_0,p_1),(q_0,q_1), prs_volcomp
    
    
def flow_torque_limiter(p0,p1,shaft_speed,torque_limit):
    #torque limiter constant power curve with 100 data points
    
    pressure = [p0,p1]
    pressure.extend(np.linspace(math.ceil(p0),math.floor(p1),100))
    pressure.sort()
    
    flow = []
    
    for p in pressure:
        q= 20.* math.pi* shaft_speed* torque_limit / (1000.0 * p)
        flow.append(q)
    
    return pressure,flow
    

def start_limiter(shaft_speed,limit_speed,pump_disp,torque_limit,coeff_leakage):
    #quadratic solution of torque limiter coincidence with vol_effy

    a=-coeff_leakage
    b=-q_theoretical(shaft_speed,pump_disp)
    c= 20.* math.pi* limit_speed* torque_limit / (1000.0 * 1.)

    x1,x2 = quadratic_solver(a,b,c)

    if 0 <= x1 <=400:
        x = x1
    elif 0 <= x2 <=400:
        x = x2
    else:
        print ('no quadratic solution')
        
    prs_start_limiter = x

    return prs_start_limiter


def stop_limiter(p_comp_slope,p_comp_intercept,torque_limit):
    #quadratic solution of torque limiter coincidence with pump compensator

    a=p_comp_slope
    b=p_comp_intercept
    c= -20.0* math.pi* float(text_nmset.text)* torque_limit /( 1000.0)

    x1,x2 = quadratic_solver(a,b,c)

    if 300 <= x1 <=400:
        x = x1
    elif 0 <= x2 <=400:
        x = x2
    else:
        print ('no quadratic solution')

    prs_stop_limiter = x
    
    return prs_stop_limiter


def max_flow_pump(label):
    #max flow warning zone derived from datasheet max rpm and pump displacement
    
    new_disp = int(radios_pumpdisp.value_selected)
    
    if new_disp == 60:
        max_flow = 60 * 2500. / 1000.
        
    elif new_disp == 110:
        max_flow = 110 * 2200. / 1000.
    
    elif new_disp ==130:
        max_flow = 130 * 2100. / 1000.
    
    else:
        max_flow = 190 * 2100. / 1000.
    
    
    max_flow_x =[0,400,400,0]
    max_flow_y =[max_flow,max_flow,440,440]
    max_flow_fill[0].set_xy(np.array([max_flow_x,max_flow_y,]).T)
    
    
#utility functions

def quadratic_solver(a,b,c):
    d = (b**2) - (4*a*c) # discriminant

    x1 = (-b+math.sqrt(b**2- 4*a*c))/(2*a)
    x2 = (-b-math.sqrt(b**2- 4*a*c))/(2*a)
    
    return x1,x2
    
    
    
#initialise pump performance data and plots from initial pump parameters


#pump limits
line_p_comp_max = ax_pq.axvline(prs_pump,linestyle='--',color='grey',linewidth=1,alpha=0.5,)
ax_pq.axvspan(400,420,color='salmon',alpha=0.1)

max_flow_x =[0,400,400,0]
max_flow_y =[130 * 2100. / 1000.,130 * 2100. / 1000.,440,440]
max_flow_fill = ax_pq.fill(max_flow_x,max_flow_y,color='salmon',alpha=0.1,zorder=0)


#pump peformance calculations

#ideal pump flow 
line_flowtheoretical = ax_pq.axhline(q_theoretical(n_input,disp_pump),linestyle='--',color='grey',linewidth=1,alpha=0.5)

#compensator flow
temp_prscomp,temp_flowcomp,prs_volcomp = flow_cp(prs_pump,n_input,disp_pump,prs_max,prs_comp_hyst,pump_leakage_coefficient)
line_flowcomp = ax_pq.plot(temp_prscomp,temp_flowcomp, color='steelblue',zorder=4)

p_comp_slope, p_comp_intercept = flow_cp(prs_pump,n_input,disp_pump,prs_max,prs_comp_hyst,pump_leakage_coefficient, True)

temp_prsfunc,temp_flowfunc,_ = flow_cp(prs_pump-prs_deltap,n_input,disp_pump,prs_max-prs_deltap,prs_comp_hyst,pump_leakage_coefficient)
line_flowfunction = ax_pq.plot(temp_prsfunc,temp_flowfunc,'--', color='green',zorder=4,alpha=0.35)


#volumetric flow
temp_prsvol,temp_flowvol = flow_act(0,prs_volcomp,n_input,disp_pump,pump_leakage_coefficient)
line_flowactual = ax_pq.plot(temp_prsvol,temp_flowvol,color='steelblue',zorder=4)


#corner power reference point
marker_cornerpower = ax_pq.plot(prs_pump,q_theoretical(n_input,disp_pump),'o',color='grey',markersize=4,alpha=1)

#
hydro_power = prs_pump * q_theoretical(n_input,disp_pump) /600.0
shaft_power = torque_limit * n_input / 9550.0

shaft_kW =  fig.text(0.025,0.70,'%5.1f'%shaft_power+' Kw',color='grey')
hydro_kW =  fig.text(0.025,0.80,'%5.1f'%hydro_power+' Kw',color='grey')

lmin_max = q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient)
max_lmin =  fig.text(0.025,0.55,'%5.1f'%(lmin_max)+' l/min',color='green')


if shaft_power < hydro_power:

    prs_start_limiter = start_limiter(n_input,n_torquelimiter_set,disp_pump,torque_limit,pump_leakage_coefficient)

    prs_stop_limiter = stop_limiter(p_comp_slope, p_comp_intercept, torque_limit)

    temp_prstorque,temp_flowtorque = flow_torque_limiter(prs_start_limiter,prs_stop_limiter,n_torquelimiter_set,torque_limit)
    line_flowtorquelimiter = ax_pq.plot(temp_prstorque,temp_flowtorque,'salmon',linewidth=2)
    
    
    
    text_minLmin =  fig.text(0.025,0.45,'%5.1f'%(line_flowtorquelimiter[0].get_ydata()[-1])+'l/min',color='orange')
    text_minPct = fig.text(0.025,0.4,'%4.1f'% ( line_flowtorquelimiter[0].get_ydata()[-1] / line_flowactual[0].get_ydata()[0] * 100. ) +' % of Max',color='grey')
    comp_start = fig.text(0.025,0.3,'%4.1f'% (prs_start_limiter-prs_deltap) +' - '+ '%4.1f'% (prs_stop_limiter-prs_deltap)+ ' bar',color='orange')
    
    fill_x=[(line_flowactual[0].get_xdata()[-1])]
    fill_x.extend(temp_prstorque)
    fill_x.append( (line_flowactual[0].get_xdata()[-1]) )
    
    fill_y=[(line_flowactual[0].get_ydata()[-1])]
    fill_y.extend(temp_flowtorque)
    fill_y.append( (line_flowactual[0].get_ydata()[-1]))

    fill_torquelimiter = ax_pq.fill(fill_x,fill_y,color='salmon',alpha=0.5,zorder=0)
    
    pressure_function = [prs_deltap,prs_deltap]
    pressure_function.extend([p-prs_deltap for p in temp_prstorque])
    pressure_function.append(prs_max-prs_deltap)
    temp_flowtorque.append(0.0)
    temp_flowtorque.insert(0,q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient))
    temp_flowtorque.insert(0,0)
    line_flowtorquelimiterfunction = ax_pq.plot(pressure_function,temp_flowtorque,'green',linewidth=2,zorder=4)
    
    deltap_noflowmax_x = [val for val in pressure_function]
    deltap_noflowmax_x.extend([prs_max-prs_deltap,prs_max])
    temp_prstorque.reverse()
    deltap_noflowmax_x.extend(temp_prstorque)
    deltap_noflowmax_x.append(pressure_function[0])
    
    deltap_noflowmax_y = [val for val in temp_flowtorque]
    deltap_noflowmax_y.extend([0,0])
    temp_flowtorque.reverse()
    deltap_noflowmax_y.extend(temp_flowtorque[1:-2])
    deltap_noflowmax_y.append(temp_flowtorque[-2])
    
    
    deltap_noflowmax = ax_pq.fill(deltap_noflowmax_x,deltap_noflowmax_y,color='salmon',alpha=0.35,fill=False,hatch='\\//')
    
    
    flow_zone_x = [prs_deltap, prs_start_limiter-prs_deltap, prs_start_limiter-prs_deltap,prs_deltap,prs_deltap]    
    flow_zone_y = [0,0,temp_flowtorque[-3],q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient),0]
    flow_zone_fill = ax_pq.fill(flow_zone_x,flow_zone_y,color='yellowgreen',alpha=0.2,zorder=0)
    

    limit_zone_x = [prs_start_limiter-prs_deltap,prs_max-prs_deltap]
    pressure_function.reverse()
    limit_zone_x.extend(pressure_function[2:-2])
    limit_zone_x.append(prs_start_limiter-prs_deltap)
    
    limit_zone_y = [0,0]
    limit_zone_y.extend(temp_flowtorque[2:-2])
    limit_zone_y.append(0)
    limit_zone_fill = ax_pq.fill(limit_zone_x,limit_zone_y,color='orange',alpha=0.2,zorder=0)
    
    
temp_prstorque,temp_flowtorque = flow_torque_limiter(1,400,n_torquelimiter_set,torque_limit)
line_torquelimiter_ref = ax_pq.plot(temp_prstorque,temp_flowtorque,'--',color='grey',linewidth=1,alpha=0.2,)


deltap_noflow_x = [0,prs_deltap,prs_deltap,0]
deltap_noflow_y = [0,0,q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient),q_theoretical(n_input,disp_pump)]

deltap_noflow_fill = ax_pq.fill(deltap_noflow_x,deltap_noflow_y,color='salmon',alpha=0.5,fill=False,hatch='\\//')


#
plt.show()
