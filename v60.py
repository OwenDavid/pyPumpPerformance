#v60.py pump performance model
#copyright Owen David 2021

import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox


#inital pump parameters

disp_pump = 130.0 ##cm3/rev
n_input = 2200 #rpm
n_torquelimiter_set = 2200 #rpm - set on dyno
prs_pump = 360 #bar
torque_limit = 455 #nm
prs_relief = 380 #bar
prs_comp_hyst = 4.0 #bar
prs_deltap = 27 #bar


#constant pump parameters

pump_leakage_coefficient = -0.04410281962 #(l/min)/bar

# results from inline hydraulik testing




#derived constants

prs_max =min(prs_pump,prs_relief)



#setup figure and axes
fig, ax_pq = plt.subplots(1, figsize=(12,6), dpi=100, facecolor='#edf1f2')
plt.subplots_adjust(left=0.25, bottom=0.3)  

#setup input widgets
axrpm= plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='#f2f2f2')
axcomp= plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='#f2f2f2')
axnm= plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='#f2f2f2')
axbox = plt.axes([0.25, 0.005, 0.04, 0.04])
axdeltap =plt.axes([0.45, 0.005, 0.03, 0.04])
pump_disp_ax = fig.add_axes([0.025, 0.06, 0.05, 0.12], facecolor='#f2f2f2')
reset_button_ax = plt.axes([0.025, 0.01, 0.05, 0.04])


reset_button = Button(reset_button_ax, 'Reset', color='#f2f2f2', hovercolor='0.975')
color_radios = RadioButtons(pump_disp_ax, ('60', '130', '190'), active=1,activecolor='steelblue')
text_box = TextBox(axbox, 'Torque Limiter (rpm)   ', initial=str(n_torquelimiter_set))
text_deltap = TextBox(axdeltap, 'Standby Pressure (bar)   ', initial=str(prs_deltap))
srpm = Slider(axrpm, 'Shaft Speed (rpm)', 0, 2500, valinit=n_input, valstep=2, color='steelblue',alpha=0.5)
scomp = Slider(axcomp, 'Pump Comp (bar)', 0, 400, valinit=prs_pump, valstep=1,color='steelblue',alpha=0.3)
snm = Slider(axnm, 'Torque Limiter (Nm)', 5, 1000, valinit=torque_limit, valstep=5,color='steelblue',alpha=0.15)


#set data field labels
shaft_pow = fig.text(0.025,0.75,'Shaft Power Limit:',color='darkgrey')
hydro_pow = fig.text(0.025,0.85,'Hydro Corner Power:',color='darkgrey')
Max_flow = fig.text(0.025,0.6,'Max Flow:',color='darkgrey')
Min_flow = fig.text(0.025,0.5,'Min Flow:',color='darkgrey')
comp_bar = fig.text(0.025,0.35,'Torque Limiting range:',color='darkgrey')


#set color scheme
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



#update pump performance after parameter change
def update(val):
    
    new_rpm = srpm.val
    if scomp.val > 400:
        new_comp = 400
    else:
        new_comp = scomp.val
    new_disp = int(color_radios.value_selected)
    n_torquelimiter_set = float(text_box.text)
    new_delta =  float(text_deltap.text)

    
    disp_pump = int(color_radios.value_selected)
    p_comp_max.set_xdata(scomp.val)
    flow_theoretical.set_ydata(q_theoretical(srpm.val,disp_pump))
    dot_ideal[0].set_data(scomp.val,q_theoretical(srpm.val,disp_pump))
    
    result3,result4,p_volcomp = flow_cp(scomp.val,srpm.val,int(color_radios.value_selected),scomp.val,prs_comp_hyst,pump_leakage_coefficient)
    flow_comp[0].set_data(result3,result4)
    
    p_comp_slope, p_comp_intercept = flow_cp(scomp.val,srpm.val,int(color_radios.value_selected),scomp.val,prs_comp_hyst,pump_leakage_coefficient, True)
    
    result1,result2 = flow_act(0,p_volcomp,srpm.val,int(color_radios.value_selected),pump_leakage_coefficient)
    flow_actual[0].set_data(result1,result2)

    
    hydro_power = (flow_actual[0].get_xdata()[-1]) * (flow_actual[0].get_ydata()[-1]) / 600.
    #q_theoretical(srpm.val,int(color_radios.value_selected)) /600.0
    shaft_power = snm.val * n_torquelimiter_set / 9550.0
    
    hydro_kW.set_text('%5.1f'%hydro_power+' Kw')
    shaft_kW.set_text('%5.1f'%shaft_power+' Kw')
    
    lmin_max = q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient)
    Max_lmin.set_text('%5.1f'%(lmin_max)+' l/min')
    
    result5,result6,_ = flow_cp(new_comp-new_delta,new_rpm,new_disp,new_comp-new_delta,prs_comp_hyst,pump_leakage_coefficient)
    flow_function[0].set_data(result5,result6)
    
    if shaft_power < hydro_power:
        
        p_start_limiter = start_limiter(srpm.val,n_torquelimiter_set,int(color_radios.value_selected),snm.val,pump_leakage_coefficient)
        p_stop_limiter = stop_limiter(p_comp_slope, p_comp_intercept, snm.val)
        pressure,flow = flow_torque_limiter(p_start_limiter,p_stop_limiter,n_torquelimiter_set,snm.val)
        flow_torque[0].set_data(pressure,flow) 
        
        
        Min_lmin.set_text('%5.1f'%(flow_torque[0].get_ydata()[-1])+' l/min')
        Min_pct.set_text('%4.1f'% ( flow_torque[0].get_ydata()[-1] / flow_actual[0].get_ydata()[0] * 100. ) +' % of Max')
        Min_lmin.set_color('orange')
        comp_start.set_text('%4.1f'% (p_start_limiter-new_delta) +' - '+ '%4.1f'% (p_stop_limiter-new_delta)+ ' bar')
        comp_start.set_color('orange')
        
        
        fill_x=[(flow_actual[0].get_xdata()[-1])]
        fill_x.extend(pressure)
        fill_x.append( (flow_actual[0].get_xdata()[-1]) )

        fill_y=[(flow_actual[0].get_ydata()[-1])]
        fill_y.extend(flow)
        fill_y.append( (flow_actual[0].get_ydata()[-1]))

        power_fill[0].set_xy(np.array([fill_x,fill_y]).T)
        power_fill[0].set_visible(True)
        
        
        pressure_function = [new_delta,new_delta]
        pressure_function.extend([p-new_delta for p in pressure])
        pressure_function.append(new_comp-new_delta)
        flow_functions = [f for f in flow]
        flow_functions.append(0.0)
        flow_functions.insert(0,q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient))
        flow_functions.insert(0,0)
        
        flow_torque_function[0].set_data(pressure_function,flow_functions)
        
        
        deltap_noflowmax_x = [val for val in pressure_function[2:-1]]
        deltap_noflowmax_x.extend([new_comp-new_delta])
        deltap_noflowmax_x.extend([new_comp])
        new_pressure = [p for p in pressure]
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
       
       
        flow_zone_x = [new_delta, p_start_limiter-new_delta, p_start_limiter-new_delta,new_delta,new_delta]    
        flow_zone_y = [0,0,fflow[-2],q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient),0]
       
        flow_zone_fill[0].set_xy(np.array([flow_zone_x,flow_zone_y]).T)
       
        
        
        limit_zone_x = [p_start_limiter-new_delta,new_comp-new_delta]
        new_pressure_function = [p for p in pressure_function[2:-1]]
        new_pressure_function.reverse()
        limit_zone_x.extend(new_pressure_function)
        limit_zone_x.append(p_start_limiter-new_delta)
        
        limit_zone_y = [0,0]
        limit_zone_y.extend(fflow)
        limit_zone_y.append(0)
        limit_zone_fill[0].set_xy(np.array([limit_zone_x,limit_zone_y,]).T)
        limit_zone_fill[0].set_visible(True)
       
        fig.canvas.draw_idle()
        
    else:
        
        flow_torque[0].set_data([],[])
        flow_torque_function[0].set_data([],[])
        power_fill[0].set_visible(False)
        
        Min_lmin.set_text('%5.1f'%(flow_actual[0].get_ydata()[-1])+' l/min')
        Min_pct.set_text('%4.1f'% ( flow_actual[0].get_ydata()[-1] / flow_actual[0].get_ydata()[0] * 100. ) +' % of Max')
        Min_lmin.set_color('green')
        comp_start.set_text('Torque Limiter not active')
        comp_start.set_color('green')
        
        result5,result6,_ = flow_cp(new_comp-new_delta,new_rpm,new_disp,new_comp-new_delta,prs_comp_hyst,pump_leakage_coefficient)
        flow_function[0].set_data(result5,result6)
        
        
        deltap_noflowmax_x = [new_comp-new_delta,new_comp,flow_actual[0].get_xdata()[-1] ,flow_function[0].get_xdata()[-1] ]
        
        deltap_noflowmax_y = [0,0,flow_actual[0].get_ydata()[-1],flow_function[0].get_ydata()[-1]]
        
        deltap_noflowmax[0].set_xy(np.array([deltap_noflowmax_x,deltap_noflowmax_y]).T)
        deltap_noflowmax[0].set_visible(True)        
        
        flow_zone_x = [new_delta, new_comp-new_delta, flow_function[0].get_xdata()[-1],new_delta,new_delta]
        
        flow_zone_y = [0,0,flow_function[0].get_ydata()[-1],q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient),0]
        
        flow_zone_fill[0].set_xy(np.array([flow_zone_x,flow_zone_y]).T)
        
        flow_torque_function[0].set_data(flow_zone_x[1:],flow_zone_y[1:])
        
        
        limit_zone_fill[0].set_visible(False)
        
        fig.canvas.draw_idle()
        
        
        
        
    
    #power_fill[0].set_xy(np.array([],[]).T)
    pressure,flow = flow_torque_limiter(1,400,n_torquelimiter_set,snm.val)
    flow_torque_ref[0].set_data(pressure,flow)
    
    deltap_noflow_x = [0,new_delta,new_delta,0]
    deltap_noflow_y = [0,0,q_actual(new_rpm,new_disp,new_delta,pump_leakage_coefficient),q_theoretical(new_rpm,new_disp)]
    
    deltap_noflow_fill[0].set_xy(np.array([deltap_noflow_x,deltap_noflow_y]).T)
    
    result5,result6,_ = flow_cp(new_comp-new_delta,new_rpm,new_disp,new_comp-new_delta,prs_comp_hyst,pump_leakage_coefficient)
    flow_function[0].set_data(result5,result6)
    
    
    
    
    fig.canvas.draw_idle()


def reset_button_on_clicked(mouse_event):
    srpm.reset()
    scomp.reset()
    snm.reset()
    color_radios.set_active(1)
    
    dot_ideal[0].set_data(prs_pump,q_theoretical(n_input,disp_pump))
    
    fig.canvas.draw_idle()   


def color_radios_on_clicked(label):
    if color_radios.value_selected == '190':
        snm.set_val(895)
        srpm.set_val(1900)
        scomp.set_val(360)
    if color_radios.value_selected == '130':
        snm.set_val(455)
        srpm.set_val(2200)
        scomp.set_val(360)
    update(label)
    max_flow_pump(label)
    fig.canvas.draw_idle()


def onclick(event):
    
    if event.inaxes in [ax_pq] and event.xdata <=400:
        if event.button == 3:
            h_power = event.xdata * event.ydata / 600.
            new_torque = h_power / float(text_box.text) * 9550.0
            snm.set_val(new_torque)
        
        elif event.button == 1:
            scomp.set_val(round(event.xdata))
            new_flow = round(event.ydata)
            new_disp = int(color_radios.value_selected)
            new_rpm = new_flow * 1000.0 / new_disp
            srpm.set_val(new_rpm)

    
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
text_box.on_submit(update)
text_deltap.on_submit(update)
srpm.on_changed(update)
scomp.on_changed(update)
snm.on_changed(update)
reset_button.on_clicked(reset_button_on_clicked)
color_radios.on_clicked(color_radios_on_clicked)

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

    p_volcomp = ( p_comp_intercept - coeff_leakage_intercept ) / (coeff_leakage - p_comp_slope)

    q_0 = p_comp_slope*p_0 + p_comp_intercept
    p_1 = p_0 - prs_comp_hyst
    q_1 = p_comp_slope*p_1 + p_comp_intercept
    
    if coeffs == True:
        return p_comp_slope, p_comp_intercept
    else:
        return (p_0,p_1),(q_0,q_1), p_volcomp
    
    
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
        
    p_start_limiter = x

    return p_start_limiter


def stop_limiter(p_comp_slope,p_comp_intercept,torque_limit):
    #quadratic solution of torque limiter coincidence with pump compensator

    a=p_comp_slope
    b=p_comp_intercept
    c= -20.0* math.pi* float(text_box.text)* torque_limit /( 1000.0)

    x1,x2 = quadratic_solver(a,b,c)

    if 300 <= x1 <=400:
        x = x1
    elif 0 <= x2 <=400:
        x = x2
    else:
        print ('no quadratic solution')

    p_stop_limiter = x
    
    return p_stop_limiter


def max_flow_pump(label):
    #max flow warning zone derived from datasheet max rpm and pump displacement
    
    disp_pump = int(color_radios.value_selected)
    
    if disp_pump == 60:
        max_flow = 60 * 2500. / 1000.
    elif disp_pump ==130:
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
p_comp_max = ax_pq.axvline(prs_pump,linestyle='--',color='grey',linewidth=1,alpha=0.5,)
ax_pq.axvspan(400,420,color='salmon',alpha=0.1)

max_flow_x =[0,400,400,0]
max_flow_y =[130 * 2100. / 1000.,130 * 2100. / 1000.,440,440]
max_flow_fill = ax_pq.fill(max_flow_x,max_flow_y,color='salmon',alpha=0.1,zorder=0)

#pump peformance

#ideal pump flow 
flow_theoretical = ax_pq.axhline(q_theoretical(n_input,disp_pump),linestyle='--',color='grey',linewidth=1,alpha=0.5,)

#compensator flow
result3,result4,p_volcomp = flow_cp(prs_pump,n_input,disp_pump,prs_max,prs_comp_hyst,pump_leakage_coefficient)
flow_comp = ax_pq.plot(result3,result4, color='steelblue',zorder=4)

p_comp_slope, p_comp_intercept = flow_cp(prs_pump,n_input,disp_pump,prs_max,prs_comp_hyst,pump_leakage_coefficient, True)

result5,result6,_ = flow_cp(prs_pump-prs_deltap,n_input,disp_pump,prs_max-prs_deltap,prs_comp_hyst,pump_leakage_coefficient)
flow_function = ax_pq.plot(result5,result6,'--', color='green',zorder=4,alpha=0.35)


#volumetric flow
result1,result2 = flow_act(0,p_volcomp,n_input,disp_pump,pump_leakage_coefficient)
flow_actual = ax_pq.plot(result1,result2,color='steelblue',zorder=4)

#result1_function = [r-prs_deltap for r in result1]
#flow_actual_function = ax_pq.plot(result1_function,result2,color='green',zorder=4)

#corner power reference point
dot_ideal = ax_pq.plot(prs_pump,q_theoretical(n_input,disp_pump),'o',color='grey',markersize=4,alpha=1)

#
hydro_power = prs_pump * q_theoretical(n_input,disp_pump) /600.0
shaft_power = torque_limit * n_input / 9550.0

shaft_kW =  fig.text(0.025,0.70,'%5.1f'%shaft_power+' Kw',color='grey')
hydro_kW =  fig.text(0.025,0.80,'%5.1f'%hydro_power+' Kw',color='grey')

lmin_max = q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient)
Max_lmin =  fig.text(0.025,0.55,'%5.1f'%(lmin_max)+' l/min',color='green')


if shaft_power < hydro_power:

    p_start_limiter = start_limiter(n_input,n_torquelimiter_set,disp_pump,torque_limit,pump_leakage_coefficient)

    p_stop_limiter = stop_limiter(p_comp_slope, p_comp_intercept, torque_limit)

    pressure,flow = flow_torque_limiter(p_start_limiter,p_stop_limiter,n_torquelimiter_set,torque_limit)
    flow_torque = ax_pq.plot(pressure,flow,'salmon',linewidth=2)
    
    
    
    Min_lmin =  fig.text(0.025,0.45,'%5.1f'%(flow_torque[0].get_ydata()[-1])+'l/min',color='orange')
    Min_pct = fig.text(0.025,0.4,'%4.1f'% ( flow_torque[0].get_ydata()[-1] / flow_actual[0].get_ydata()[0] * 100. ) +' % of Max',color='grey')
    comp_start = fig.text(0.025,0.3,'%4.1f'% (p_start_limiter-prs_deltap) +' - '+ '%4.1f'% (p_stop_limiter-prs_deltap)+ ' bar',color='orange')
    
    fill_x=[(flow_actual[0].get_xdata()[-1])]
    fill_x.extend(pressure)
    fill_x.append( (flow_actual[0].get_xdata()[-1]) )
    
    fill_y=[(flow_actual[0].get_ydata()[-1])]
    fill_y.extend(flow)
    fill_y.append( (flow_actual[0].get_ydata()[-1]))

    power_fill = ax_pq.fill(fill_x,fill_y,color='salmon',alpha=0.5,zorder=0)
    
    pressure_function = [prs_deltap,prs_deltap]
    pressure_function.extend([p-prs_deltap for p in pressure])
    pressure_function.append(prs_max-prs_deltap)
    flow.append(0.0)
    flow.insert(0,q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient))
    flow.insert(0,0)
    flow_torque_function = ax_pq.plot(pressure_function,flow,'green',linewidth=2,zorder=4)
    
    deltap_noflowmax_x = [val for val in pressure_function]
    deltap_noflowmax_x.extend([prs_max-prs_deltap,prs_max])
    pressure.reverse()
    deltap_noflowmax_x.extend(pressure)
    deltap_noflowmax_x.append(pressure_function[0])
    
    deltap_noflowmax_y = [val for val in flow]
    deltap_noflowmax_y.extend([0,0])
    flow.reverse()
    deltap_noflowmax_y.extend(flow[1:-2])
    deltap_noflowmax_y.append(flow[-2])
    
    
    deltap_noflowmax = ax_pq.fill(deltap_noflowmax_x,deltap_noflowmax_y,color='salmon',alpha=0.35,fill=False,hatch='\\//')
    
    
    flow_zone_x = [prs_deltap, p_start_limiter-prs_deltap, p_start_limiter-prs_deltap,prs_deltap,prs_deltap]    
    flow_zone_y = [0,0,flow[-3],q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient),0]
    flow_zone_fill = ax_pq.fill(flow_zone_x,flow_zone_y,color='yellowgreen',alpha=0.2,zorder=0)
    

    limit_zone_x = [p_start_limiter-prs_deltap,prs_max-prs_deltap]
    pressure_function.reverse()
    limit_zone_x.extend(pressure_function[2:-2])
    limit_zone_x.append(p_start_limiter-prs_deltap)
    
    limit_zone_y = [0,0]
    limit_zone_y.extend(flow[2:-2])
    limit_zone_y.append(0)
    limit_zone_fill = ax_pq.fill(limit_zone_x,limit_zone_y,color='orange',alpha=0.2,zorder=0)
    
    
pressure,flow = flow_torque_limiter(1,400,n_torquelimiter_set,torque_limit)
flow_torque_ref = ax_pq.plot(pressure,flow,'--',color='grey',linewidth=1,alpha=0.2,)


deltap_noflow_x = [0,prs_deltap,prs_deltap,0]
deltap_noflow_y = [0,0,q_actual(n_input,disp_pump,prs_deltap,pump_leakage_coefficient),q_theoretical(n_input,disp_pump)]

deltap_noflow_fill = ax_pq.fill(deltap_noflow_x,deltap_noflow_y,color='salmon',alpha=0.5,fill=False,hatch='\\//')


#
plt.show()
