#!/usr/bin/env pytest
###############################################################################
# $Id$
#
# Project:  GDAL/OGR Test Suite
# Purpose:  Test multidimensional support in GTiff driver
# Author:   Even Rouault <even.rouault@spatialys.com>
#
###############################################################################
# Copyright (c) 2021, Even Rouault <even.rouault@spatialys.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
###############################################################################

import array

from osgeo import gdal

import pytest

# All tests will be skipped if numpy is unavailable
numpy = pytest.importorskip('numpy')


def _test_basic_2D(blockSize, arraySize, dt):

    filename = '/vsimem/mdim.tif'

    def write():
        ds = gdal.GetDriverByName('GTiff').CreateMultiDimensional(filename)
        rg = ds.GetRootGroup()
        dimY = rg.CreateDimension("dimY", None, None, arraySize[0])
        dimX = rg.CreateDimension("dimX", None, None, arraySize[1])

        ar = rg.CreateMDArray("myarray", [dimY, dimX],
                              gdal.ExtendedDataType.Create(dt),
                              ['BLOCKSIZE=%d,%d' % (blockSize[0], blockSize[1])])
        numpy_ar = numpy.reshape(numpy.arange(0, dimY.GetSize() * dimX.GetSize(), dtype=numpy.uint16),
                                 (dimY.GetSize(), dimX.GetSize()))
        if dt == gdal.GDT_Byte:
            numpy_ar = numpy.clip(numpy_ar, 0, 255)
        assert ar.Write(numpy_ar)
        return numpy_ar

    ref_ar = write()

    ds = gdal.Open(filename)
    assert ds.GetRasterBand(1).XSize == arraySize[1]
    assert ds.GetRasterBand(1).YSize == arraySize[0]
    assert ds.GetRasterBand(1).DataType == dt
    assert ds.GetRasterBand(1).GetBlockSize() == [blockSize[1], blockSize[0]]
    assert numpy.array_equal(ds.ReadAsArray(), ref_ar)
    assert ds.GetMetadata() == {
        'DIMENSION_0_BLOCK_SIZE': '%d' % blockSize[0],
        'DIMENSION_0_NAME': 'dimY',
        'DIMENSION_0_SIZE': '%d' % arraySize[0],
        'DIMENSION_1_BLOCK_SIZE': '%d' % blockSize[1],
        'DIMENSION_1_NAME': 'dimX',
        'DIMENSION_1_SIZE': '%d' % arraySize[1],
        'VARIABLE_NAME': 'myarray'}

    gdal.Unlink(filename)


@pytest.mark.parametrize("blockSize,arraySize", [[(16, 32), (32, 64)],  # raster size exact multiple of block size
                                                 # just one truncated block
                                                 [(16, 32), (15, 31)],
                                                 # three truncated blocks
                                                 [(16, 32), (33, 65)],
                                                 ])
def test_gtiff_mdim_basic_2D_blocksize(blockSize, arraySize):
    _test_basic_2D(blockSize, arraySize, gdal.GDT_UInt16)


@pytest.mark.parametrize("datatype", [gdal.GDT_Byte,
                                      gdal.GDT_Int16,
                                      gdal.GDT_UInt16,
                                      gdal.GDT_Int32,
                                      gdal.GDT_UInt32,
                                      gdal.GDT_Float32,
                                      gdal.GDT_Float64,
                                      gdal.GDT_CInt16,
                                      gdal.GDT_CInt32,
                                      gdal.GDT_CFloat32,
                                      gdal.GDT_CFloat64, ], ids=gdal.GetDataTypeName)
def test_gtiff_mdim_basic_2D_datatype(datatype):
    _test_basic_2D((16, 32), (16, 32), datatype)


@pytest.mark.parametrize("blockSize,arraySize", [[(1, 16, 32), (1, 32, 64)],
                                                 [(1, 16, 32), (2, 32, 64)],
                                                 [(2, 16, 32), (2, 32, 64)],
                                                 [(2, 16, 32), (5, 32, 64)],
                                                 [(3, 16, 32), (2, 32, 64)]])
def test_gtiff_mdim_basic_3D(blockSize, arraySize):

    filename = '/vsimem/mdim.tif'
    dt = gdal.GDT_UInt16
    z_values = [i + 1 for i in range(arraySize[0])]

    def write():
        ds = gdal.GetDriverByName('GTiff').CreateMultiDimensional(filename)
        rg = ds.GetRootGroup()
        dimZ = rg.CreateDimension("dimZ", 'a', 'b', arraySize[0])
        dimY = rg.CreateDimension("dimY", None, None, arraySize[1])
        dimX = rg.CreateDimension("dimX", None, None, arraySize[2])

        dimZVar = rg.CreateMDArray("dimZ", [dimZ], gdal.ExtendedDataType.Create(
            gdal.GDT_Int32), ['IS_INDEXING_VARIABLE=YES'])
        dimZ.SetIndexingVariable(dimZVar)
        dimZVar.Write(array.array('f', z_values))

        ar = rg.CreateMDArray("myarray", [dimZ, dimY, dimX],
                              gdal.ExtendedDataType.Create(dt),
                              ['BLOCKSIZE=%d,%d,%d' % (blockSize[0], blockSize[1], blockSize[2])])
        numpy_ar = numpy.reshape(numpy.arange(0, dimZ.GetSize() * dimY.GetSize() * dimX.GetSize(), dtype=numpy.uint16),
                                 (dimZ.GetSize(), dimY.GetSize(), dimX.GetSize()))
        if dt == gdal.GDT_Byte:
            numpy_ar = numpy.clip(numpy_ar, 0, 255)
        assert ar.Write(numpy_ar)
        return numpy_ar

    ref_ar = write()

    # Iterate over IFDs
    for idx in range(arraySize[0]):
        ds = gdal.Open('GTIFF_DIR:%d:%s' % (idx+1, filename))
        assert ds.GetRasterBand(1).XSize == arraySize[-1]
        assert ds.GetRasterBand(1).YSize == arraySize[-2]
        assert ds.GetRasterBand(1).DataType == dt
        assert ds.GetRasterBand(1).GetBlockSize() == [
            blockSize[-1], blockSize[-2]]
        assert numpy.array_equal(ds.ReadAsArray(), ref_ar[idx])
        # Minimum details beyond first IFD
        expected_md = {
            'DIMENSION_0_NAME': 'dimZ',
            'DIMENSION_0_IDX': '%d' % idx,
            'DIMENSION_0_VAL': '%d' % z_values[idx],
            'VARIABLE_NAME': 'myarray'
        }
        if idx == 0:
            # Full details for first IFD
            expected_md.update({
                'DIMENSION_0_BLOCK_SIZE': '%d' % blockSize[0],
                'DIMENSION_0_DATATYPE': 'Int32',
                'DIMENSION_0_DIRECTION': 'b',
                'DIMENSION_0_SIZE': '%d' % arraySize[0],
                'DIMENSION_0_TYPE': 'a',
                'DIMENSION_0_VALUES': ','.join(str(v) for v in z_values),
                'DIMENSION_1_BLOCK_SIZE': '%d' % blockSize[1],
                'DIMENSION_1_NAME': 'dimY',
                'DIMENSION_1_SIZE': '%d' % arraySize[1],
                'DIMENSION_2_BLOCK_SIZE': '%d' % blockSize[2],
                'DIMENSION_2_NAME': 'dimX',
                'DIMENSION_2_SIZE': '%d' % arraySize[2]})
        assert ds.GetMetadata() == expected_md

    gdal.Unlink(filename)


@pytest.mark.parametrize("blockSize,arraySize", [  # [(1, 1, 16, 32), (1, 1, 32, 64)],
    [(2, 3, 16, 32), (5, 8, 32, 64)],
])
def test_gtiff_mdim_basic_4D(blockSize, arraySize):

    filename = '/vsimem/mdim.tif'
    dt = gdal.GDT_UInt16
    t_values = [i + 1 for i in range(arraySize[0])]
    z_values = [i + 1 for i in range(arraySize[1])]

    def write():
        ds = gdal.GetDriverByName('GTiff').CreateMultiDimensional(filename)
        rg = ds.GetRootGroup()
        dimT = rg.CreateDimension("dimT", None, None, arraySize[0])
        dimZ = rg.CreateDimension("dimZ", 'a', 'b', arraySize[1])
        dimY = rg.CreateDimension("dimY", None, None, arraySize[2])
        dimX = rg.CreateDimension("dimX", None, None, arraySize[3])

        dimTVar = rg.CreateMDArray("dimT", [dimT], gdal.ExtendedDataType.Create(
            gdal.GDT_Int32), ['IS_INDEXING_VARIABLE=YES'])
        dimT.SetIndexingVariable(dimTVar)
        dimTVar.Write(array.array('f', t_values))

        dimZVar = rg.CreateMDArray("dimZ", [dimZ], gdal.ExtendedDataType.Create(
            gdal.GDT_Int32), ['IS_INDEXING_VARIABLE=YES'])
        dimZ.SetIndexingVariable(dimZVar)
        dimZVar.Write(array.array('f', z_values))

        ar = rg.CreateMDArray("myarray", [dimT, dimZ, dimY, dimX],
                              gdal.ExtendedDataType.Create(dt),
                              ['BLOCKSIZE=%d,%d,%d,%d' % (blockSize[0], blockSize[1], blockSize[2], blockSize[3])])
        numpy_ar = numpy.reshape(numpy.arange(0, dimT.GetSize() * dimZ.GetSize() * dimY.GetSize() * dimX.GetSize(), dtype=numpy.uint16),
                                 (dimT.GetSize(), dimZ.GetSize(), dimY.GetSize(), dimX.GetSize()))
        if dt == gdal.GDT_Byte:
            numpy_ar = numpy.clip(numpy_ar, 0, 255)
        assert ar.Write(numpy_ar)
        return numpy_ar

    ref_ar = write()

    # Iterate over IFDs
    for idx_t in range(arraySize[0]):
        for idx_z in range(arraySize[1]):
            idx = idx_t * arraySize[1] + idx_z

            ds = gdal.Open('GTIFF_DIR:%d:%s' % (idx+1, filename))
            assert ds.GetRasterBand(1).XSize == arraySize[-1]
            assert ds.GetRasterBand(1).YSize == arraySize[-2]
            assert ds.GetRasterBand(1).DataType == dt
            assert ds.GetRasterBand(1).GetBlockSize() == [
                blockSize[-1], blockSize[-2]]
            assert numpy.array_equal(ds.ReadAsArray(), ref_ar[idx_t][idx_z])
            # Minimum details beyond first IFD
            expected_md = {
                'DIMENSION_0_NAME': 'dimT',
                'DIMENSION_0_IDX': '%d' % idx_t,
                'DIMENSION_0_VAL': '%d' % t_values[idx_t],

                'DIMENSION_1_NAME': 'dimZ',
                'DIMENSION_1_IDX': '%d' % idx_z,
                'DIMENSION_1_VAL': '%d' % z_values[idx_z],

                'VARIABLE_NAME': 'myarray'
            }
            if idx == 0:
                # Full details for first IFD
                expected_md.update({
                    'DIMENSION_0_BLOCK_SIZE': '%d' % blockSize[0],
                    'DIMENSION_0_DATATYPE': 'Int32',
                    'DIMENSION_0_SIZE': '%d' % arraySize[0],
                    'DIMENSION_0_VALUES': ','.join(str(v) for v in t_values),

                    'DIMENSION_1_BLOCK_SIZE': '%d' % blockSize[1],
                    'DIMENSION_1_DATATYPE': 'Int32',
                    'DIMENSION_1_DIRECTION': 'b',
                    'DIMENSION_1_SIZE': '%d' % arraySize[1],
                    'DIMENSION_1_TYPE': 'a',
                    'DIMENSION_1_VALUES': ','.join(str(v) for v in z_values),

                    'DIMENSION_2_BLOCK_SIZE': '%d' % blockSize[2],
                    'DIMENSION_2_NAME': 'dimY',
                    'DIMENSION_2_SIZE': '%d' % arraySize[2],

                    'DIMENSION_3_BLOCK_SIZE': '%d' % blockSize[3],
                    'DIMENSION_3_NAME': 'dimX',
                    'DIMENSION_3_SIZE': '%d' % arraySize[3]})
            assert ds.GetMetadata() == expected_md

    gdal.Unlink(filename)


def test_gtiff_mdim_array_attributes_scale_offset_nodata():

    filename = '/vsimem/mdim.tif'

    def write():
        ds = gdal.GetDriverByName('GTiff').CreateMultiDimensional(filename)
        rg = ds.GetRootGroup()
        dimY = rg.CreateDimension("dimY", None, None, 1)
        dimX = rg.CreateDimension("dimX", None, None, 1)

        ar = rg.CreateMDArray("myarray", [dimY, dimX],
                              gdal.ExtendedDataType.Create(gdal.GDT_Byte))

        att = ar.CreateAttribute(
            'att_text', [], gdal.ExtendedDataType.CreateString())
        assert att
        assert att.Write('foo') == gdal.CE_None

        att = ar.CreateAttribute(
            'att_text_null', [], gdal.ExtendedDataType.CreateString())
        assert att

        att = ar.CreateAttribute(
            'att_text_multiple', [2], gdal.ExtendedDataType.CreateString())
        assert att
        assert att.Write(['foo', 'bar']) == gdal.CE_None

        att = ar.CreateAttribute(
            'att_int', [], gdal.ExtendedDataType.Create(gdal.GDT_Int32))
        assert att
        assert att.Write(123456789) == gdal.CE_None

        att = ar.CreateAttribute(
            'att_int_multiple', [2], gdal.ExtendedDataType.Create(gdal.GDT_Int32))
        assert att
        assert att.Write([123456789, 23]) == gdal.CE_None

        assert ar.SetOffset(1.25) == gdal.CE_None
        assert ar.SetScale(3.25) == gdal.CE_None
        assert ar.SetUnit('my unit') == gdal.CE_None

        assert ar.SetNoDataValueDouble(23) == gdal.CE_None

    write()

    ds = gdal.Open(filename)
    assert ds.GetMetadataItem('att_text') == 'foo'
    assert ds.GetMetadataItem('att_text_null') == 'null'
    assert ds.GetMetadataItem('att_text_multiple') == 'foo,bar'
    assert ds.GetMetadataItem('att_int') == '123456789'
    assert ds.GetMetadataItem('att_int_multiple') == '123456789,23'
    assert ds.GetRasterBand(1).GetOffset() == 1.25
    assert ds.GetRasterBand(1).GetScale() == 3.25
    assert ds.GetRasterBand(1).GetUnitType() == 'my unit'
    assert ds.GetRasterBand(1).GetNoDataValue() == 23

    gdal.Unlink(filename)
