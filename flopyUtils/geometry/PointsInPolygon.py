#!/usr/bin/env python

def PointsInPolygon(shpfile, x, y, debug=0): # {{{
    '''
    Explain
     find points location in polygon.

    Usage
     shapefile = '*.shp' # *.shp file format data.
     pos = PointsInPolygon(shpfile,x,y)

     # use points in polygon
     xy_bc = np.array([[1,2],
                [3,4],
                [5,6]])
     pos = PointsInPolygon(xy_bc,x,y)
    '''
    import geopandas, pandas
    import shapely.geometry
    import numpy as np

    # load polygons from shpfile.
    if isinstance(shpfile,str):
        polygons = geopandas.read_file(shpfile)
        if debug:
            print('check size of polygons')
            print('   shape of geometry = {}'.format(np.shape(polygons.geometry)))
    elif isinstance(shpfile,shapely.geometry.polygon.Polygon):
        # reconstruct polygons with geopandas sytle.
        polygons = geopandas.GeoSeries(shpfile)
    elif isinstance(shpfile,geopandas.geodataframe.GeoDataFrame):
        polygons = shpfile
    else: # array type
        xy_bc = shpfile

        # get size of array
        nx, ny = np.shape(xy_bc)

        # set DataFrame of pandas
        df_poly = pandas.DataFrame(xy_bc,columns=['x','y'])
        points  = [shapely.geometry.Point(xy) for xy in zip(df_poly.x, df_poly.y)]
        poly = shapely.geometry.Polygon([(p.x, p.y) for p in points])
        polygons = geopandas.GeoSeries(poly)

    if debug:
        print('   find points in polygons.')

    if isinstance(x,float) & isinstance(y,float):
       x = [x]
       y = [y]
    s   = np.shape(x)
    npg = len(polygons.boundary) # number of polygon geometry

    if len(s) == 2:
        npts = s[0]*s[1]
        #pos = np.zeros((s[0],s[1],npg))
        #cn  = 0
        #for i in range(s[0]):
        #    for j in range(s[1]):
        #        cn += 1
        #        print('   processing: %d/%d'%(cn,npts),end='\r')
        #        p = shapely.geometry.Point(x[i,j],y[i,j])
        #        for k, pg in enumerate(polygons.geometry):
        #            tmp = pg.contains(p)
        #            pos[i,j,k] = tmp
        #print('')
        pos = np.zeros((s[0],s[1],npg))
        pos = np.reshape(pos,(npts,npg))
        p   = []
        if debug:
           print('   initialize points.')
        for i in range(s[0]):
           for j in range(s[1]):
              p.append(shapely.geometry.Point(x[i,j],y[i,j]))
        #import multiprocessing
        import parmap
        for k, pg in enumerate(polygons.geometry):
           #with multiprocessing.Pool(10) as pool:
              #pos[:,k] = pool.map(pg.contains,p)
           pos[:,k] = parmap.map(pg.contains,p, pm_pbar=True)
        pos = np.reshape(pos,(s[0],s[1],npg))
        pos = np.sum(pos,axis=2)
    else:
        pos = np.zeros(s,dtype=bool)
        for i in range(s[0]):
            p = shapely.geometry.Point(x[i],y[i])
            if np.any(polygons.contains(p)):
               pos[i] = True

    return pos
# }}}

if __name__ == '__main__': # {{{
   import numpy as np
   xy_bc = np.array([[0,0],
             [1,0],
             [1,1],
             [0,1],
             [0,0]])
   x = [.5, 1.5]
   y = [.5, 1.5]
   pos = PointsInPolygon(xy_bc,x,y)
   print(pos)
   # }}}
