import math
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from matplotlib.widgets import Slider, Button, RadioButtons


#inputs

disp_pump = 130.0 ##cm3/rev
n_input = 2200 #rpm
n_torquelimiter_set = 2200 #rpm - set on dyno

prs_pump = 360 #bar
torque_limit = 455 #nm
prs_relief = 380
prs_comp_hyst = 4.0


#constants

pump_leakage_coefficient = -0.04410281962 #(l/min)/bar


#derived constants

prs_max =min(prs_pump,prs_relief)


#setup figure and axes
fig, ax_pq = plt.subplots(1)
plt.subplots_adjust(left=0.25, bottom=0.3)    

#set axis bounds
ax_pq.set_ylim(0,440)
ax_pq.set_xlim(0,420)

#set axis labels
ax_pq.set_xlabel('P - pressure (bar)')
ax_pq.set_ylabel('Q - flow rate (l/min)')

#set plot title
ax_pq.set_title('HAWE v60n Pump Performance', color='grey')

#set grid and layout
ax_pq.grid(linestyle='--',linewidth=0.3,alpha=0.3)

plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams["figure.facecolor"] = "#edf1f2"
plt.rcParams["axes.facecolor"] = "#f2f2f2"
plt.rcParams['font.size']= 8

plt.text(15, -215, 'Torque Limit Ref Speed = '+str(n_torquelimiter_set)+' rpm',
         bbox=dict(facecolor='salmon', alpha=0.1),clip_on=False, fontsize=4, style='italic')

#update function
def update(val):
    
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
    
    Max_lmin.set_text('%5.1f'%(flow_actual[0].get_ydata()[0])+' l/min')
    
    if shaft_power < hydro_power:
        
        p_start_limiter = start_limiter(srpm.val,int(color_radios.value_selected),snm.val,pump_leakage_coefficient)
        p_stop_limiter = stop_limiter(p_comp_slope, p_comp_intercept)

        pressure,flow = flow_torque_limiter(p_start_limiter,p_stop_limiter,srpm.val,snm.val)
        flow_torque[0].set_data(pressure,flow) 
        
        Min_lmin.set_text('%5.1f'%(flow_torque[0].get_ydata()[-1])+' l/min')
        
        fill_x=[(flow_actual[0].get_xdata()[-1])]
        fill_x.extend(pressure)
        fill_x.append( (flow_actual[0].get_xdata()[-1]) )

        fill_y=[(flow_actual[0].get_ydata()[-1])]
        fill_y.extend(flow)
        fill_y.append( (flow_actual[0].get_ydata()[-1]))

        power_fill[0].set_xy(np.array([fill_x,fill_y]).T)
        power_fill[0].set_visible(True)
        
        fig.canvas.draw_idle()
        
    else:
        flow_torque[0].set_data([],[])
        power_fill[0].set_visible(False)
        
        Min_lmin.set_text('%5.1f'%(flow_actual[0].get_ydata()[-1])+' l/min')
        
        fig.canvas.draw_idle()
    
    #power_fill[0].set_xy(np.array([],[]).T)
    pressure,flow = flow_torque_limiter(1,400,srpm.val,snm.val)
    flow_torque_ref[0].set_data(pressure,flow)
    
    fig.canvas.draw_idle()


#setup input sliders
axrpm= plt.axes([0.25, 0.15, 0.65, 0.03])
axcomp= plt.axes([0.25, 0.1, 0.65, 0.03])
axnm= plt.axes([0.25, 0.05, 0.65, 0.03])

srpm = Slider(axrpm, 'Shaft Speed (rpm)', 0, 2500, valinit=n_input, valstep=2, color='steelblue',alpha=0.5)
srpm.on_changed(update)

scomp = Slider(axcomp, 'Pump Comp (bar)', 0, 400, valinit=prs_pump, valstep=1,color='steelblue',alpha=0.3)
scomp.on_changed(update)

snm = Slider(axnm, 'Torque Limiter (Nm)', 0, 1000, valinit=torque_limit, valstep=5,color='steelblue',alpha=0.15)
snm.on_changed(update)

#setup a button for resetting the parameters
reset_button_ax = plt.axes([0.025, 0.2, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color='#f2f2f2', hovercolor='0.975')

def reset_button_on_clicked(mouse_event):
    srpm.reset()
    scomp.reset()
    snm.reset()
    color_radios.set_active(1)
    
    dot_ideal[0].set_data(prs_pump,q_theoretical(n_input,disp_pump))
    
    fig.canvas.draw_idle()   
reset_button.on_clicked(reset_button_on_clicked)


pump_disp_ax = fig.add_axes([0.025, 0.25, 0.1, 0.12], facecolor='#f2f2f2')
color_radios = RadioButtons(pump_disp_ax, ('60', '130', '190'), active=1)
def color_radios_on_clicked(label):
    update(label)
    max_flow_pump(label)
    fig.canvas.draw_idle()
color_radios.on_clicked(color_radios_on_clicked)



#utility functions

def quadratic_solver(a,b,c):
    d = (b**2) - (4*a*c) # discriminant

    x1 = (-b+math.sqrt(b**2- 4*a*c))/(2*a)
    x2 = (-b-math.sqrt(b**2- 4*a*c))/(2*a)
    
    return x1,x2


#functions

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
        q= 20.* math.pi* n_torquelimiter_set* torque_limit / (1000.0 * p)
        flow.append(q)
    
    return pressure,flow
    

def start_limiter(shaft_speed,pump_disp,torque_limit,coeff_leakage):
    
    #quadratic solution of torque limiter coincidence with vol_effy

    a=-coeff_leakage
    b=-q_theoretical(shaft_speed,pump_disp)
    c= 20.* math.pi* n_input* torque_limit / (1000.0 * 1.)

    x1,x2 = quadratic_solver(a,b,c)

    if 0 <= x1 <=400:
        x = x1
    elif 0 <= x2 <=400:
        x = x2
    else:
        print 'no quadratic solution'

    p_start_limiter = x

    return p_start_limiter


def stop_limiter(p_comp_slope,p_comp_intercept):
    
    #quadratic solution of torque limiter coincidence with pump compensator

    a=p_comp_slope
    b=p_comp_intercept
    c= -20.0*math.pi*n_torquelimiter_set*torque_limit/1000.0

    x1,x2 = quadratic_solver(a,b,c)

    if 300 <= x1 <=400:
        x = x1
    elif 0 <= x2 <=400:
        x = x2
    else:
        print 'no quadratic solution'

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
    
    
#initialise plot components with 'reset' data values. representative setup from Cariboni

p_comp_max = ax_pq.axvline(prs_pump,linestyle='--',color='grey',linewidth=1,alpha=0.5,)

flow_theoretical = ax_pq.axhline(q_theoretical(n_input,disp_pump),linestyle='--',color='grey',linewidth=1,alpha=0.5,)


result3,result4,p_volcomp = flow_cp(prs_pump,n_input,disp_pump,prs_max,prs_comp_hyst,pump_leakage_coefficient)
flow_comp = ax_pq.plot(result3,result4, color='steelblue',zorder=4)

p_comp_slope, p_comp_intercept = flow_cp(prs_pump,n_input,disp_pump,prs_max,prs_comp_hyst,pump_leakage_coefficient, True)

result1,result2 = flow_act(0,p_volcomp,n_input,disp_pump,pump_leakage_coefficient)
flow_actual = ax_pq.plot(result1,result2,color='steelblue',zorder=4)

dot_ideal = ax_pq.plot(prs_pump,q_theoretical(n_input,disp_pump),'o',color='grey',markersize=4,alpha=1)


hydro_power = prs_pump * q_theoretical(n_input,disp_pump) /600.0
shaft_power = torque_limit * n_input / 9550.0

if shaft_power < hydro_power:

    p_start_limiter = start_limiter(n_input,disp_pump,torque_limit,pump_leakage_coefficient)

    p_stop_limiter = stop_limiter(p_comp_slope, p_comp_intercept)

    pressure,flow = flow_torque_limiter(p_start_limiter,p_stop_limiter,n_input,torque_limit)
    flow_torque = ax_pq.plot(pressure,flow,'salmon',linewidth=2)
    
    
    fill_x=[(flow_actual[0].get_xdata()[-1])]
    fill_x.extend(pressure)
    fill_x.append( (flow_actual[0].get_xdata()[-1]) )
    
    fill_y=[(flow_actual[0].get_ydata()[-1])]
    fill_y.extend(flow)
    fill_y.append( (flow_actual[0].get_ydata()[-1]))

    power_fill = ax_pq.fill(fill_x,fill_y,color='salmon',alpha=0.5,zorder=0)
  

pressure,flow = flow_torque_limiter(1,400,n_input,torque_limit)
flow_torque_ref = ax_pq.plot(pressure,flow,'--',color='grey',linewidth=1,alpha=0.2,)

ax_pq.axvspan(400,420,color='salmon',alpha=0.05)

max_flow_x =[0,400,400,0]
max_flow_y =[130 * 2100. / 1000.,130 * 2100. / 1000.,440,440]

max_flow_fill = ax_pq.fill(max_flow_x,max_flow_y,color='salmon',alpha=0.05,zorder=0)


#derived data updated on parameter change

shaft_pow = fig.text(0.025,0.75,'Shaft Power:',color='grey')
shaft_kW =  fig.text(0.025,0.70,'%5.1f'%shaft_power+' Kw',color='grey')
hydro_pow = fig.text(0.025,0.85,'Hydro Power:',color='grey')
hydro_kW =  fig.text(0.025,0.80,'%5.1f'%hydro_power+' Kw',color='grey')

Max_flow = fig.text(0.025,0.6,'Max Flow:',color='grey')
Max_lmin =  fig.text(0.025,0.55,'%5.1f'%(flow_actual[0].get_ydata()[0])+'l/min',color='grey')
Min_flow = fig.text(0.025,0.5,'Min Flow:',color='grey')
Min_lmin =  fig.text(0.025,0.45,'%5.1f'%(flow_torque[0].get_ydata()[-1])+'l/min',color='grey')

#
plt.show()
