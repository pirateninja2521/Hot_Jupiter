import mesa_reader as mr

g = mr.MesaLogDir()
p = g.profile_data()
radius_p = p.data('radius')
print(radius_p)