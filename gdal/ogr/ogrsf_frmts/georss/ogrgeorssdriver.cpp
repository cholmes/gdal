/******************************************************************************
 * $Id$
 *
 * Project:  GeoRSS Translator
 * Purpose:  Implements OGRGeoRSSDriver.
 * Author:   Even Rouault, even dot rouault at mines dash paris dot org
 *
 ******************************************************************************
 * Copyright (c) 2008, Even Rouault <even dot rouault at mines-paris dot org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************/

#include "ogr_georss.h"
#include "cpl_conv.h"

CPL_CVSID("$Id$");

/************************************************************************/
/*                                Open()                                */
/************************************************************************/

static GDALDataset *OGRGeoRSSDriverOpen( GDALOpenInfo* poOpenInfo )

{
    if( poOpenInfo->eAccess == GA_Update || poOpenInfo->fpL == NULL )
        return NULL;

    if( strstr((const char*)poOpenInfo->pabyHeader, "<rss") == NULL &&
        strstr((const char*)poOpenInfo->pabyHeader, "<feed") == NULL )
        return NULL;

    OGRGeoRSSDataSource   *poDS = new OGRGeoRSSDataSource();

    if( !poDS->Open( poOpenInfo->pszFilename, poOpenInfo->eAccess == GA_Update ) )
    {
        delete poDS;
        poDS = NULL;
    }

    return poDS;
}

/************************************************************************/
/*                               Create()                               */
/************************************************************************/

static GDALDataset *OGRGeoRSSDriverCreate( const char * pszName,
                                    int nBands, int nXSize, int nYSize, GDALDataType eDT,
                                    char **papszOptions )
{
    OGRGeoRSSDataSource   *poDS = new OGRGeoRSSDataSource();

    if( !poDS->Create( pszName, papszOptions ) )
    {
        delete poDS;
        poDS = NULL;
    }

    return poDS;
}

/************************************************************************/
/*                               Delete()                               */
/************************************************************************/

static CPLErr OGRGeoRSSDriverDelete( const char *pszFilename )

{
    if( VSIUnlink( pszFilename ) == 0 )
        return CE_None;
    else
        return CE_Failure;
}

/************************************************************************/
/*                           RegisterOGRGeoRSS()                           */
/************************************************************************/

void RegisterOGRGeoRSS()

{
    if (! GDAL_CHECK_VERSION("OGR/GeoRSS driver"))
        return;
    GDALDriver  *poDriver;

    if( GDALGetDriverByName( "GeoRSS" ) == NULL )
    {
        poDriver = new GDALDriver();

        poDriver->SetDescription( "GeoRSS" );
        poDriver->SetMetadataItem( GDAL_DCAP_VECTOR, "YES" );
        poDriver->SetMetadataItem( GDAL_DMD_LONGNAME,
                                   "GeoRSS" );
        poDriver->SetMetadataItem( GDAL_DMD_HELPTOPIC,
                                   "drv_georss.html" );

        poDriver->SetMetadataItem( GDAL_DCAP_VIRTUALIO, "YES" );

        poDriver->pfnOpen = OGRGeoRSSDriverOpen;
        poDriver->pfnCreate = OGRGeoRSSDriverCreate;
        poDriver->pfnDelete = OGRGeoRSSDriverDelete;

        GetGDALDriverManager()->RegisterDriver( poDriver );
    }
}

