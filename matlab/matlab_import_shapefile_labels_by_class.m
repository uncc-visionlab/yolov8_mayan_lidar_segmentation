clear;
clc;

hold off;
close all;

DATASETINDEX = 4;
PLOT_REGION_MESH = false;
MAX_INTENSITY = 255;

object = 'Annular structure';
if strcmp(object, 'Annular structure')
    region(1).Name = 'Annular structure';
    region(1).WINDOWSIZE = 40;
    region(1).Color = [1 .8 .8]; % light red
    region(1).LabelValue = 1;   % label value for both annular and plarform are 1 since we create seperate datasets due to the imbalanced dataset size
    KOM_label_data = "geo_raw_data/KOM/Labels/Kom_annulars_outline.shp";
    MLS_label_data = "geo_raw_data/MLS/Labels_All/MLS_Annulars_outline_v2.shp";
    SAY_clipped_label_data = "geo_raw_data/SAY_clipped/Labels/SAY_annular_polygons.shp";
    HNT_label_data = "geo_raw_data/HNT/Labels/HNT_Annular_OTLs_2023.shp";

    KOM_matlab_gt_labels_all_filename = "KOM/KOM_gt_annular.mat";
    MLS_matlab_gt_labels_all_filename = "MLS/MLS_gt_annular.mat";
    SAY_clipped_matlab_gt_labels_all_filename = "SAY_clipped/SAY_clipped_gt_annular.mat";
    HNT_matlab_gt_labels_all_filename = "HNT/HNT_4-5km2_gt_annular.mat";
else
    region(1).Name = 'Platform';
    region(1).WINDOWSIZE = 80;
    region(1).Color = [.67 .84 .9]; % light blue
    region(1).LabelValue = 1;
    
    KOM_label_data = "geo_raw_data/KOM/Labels/Kom_new_platform_training_polygons.shp";
    MLS_label_data = "geo_raw_data/MLS/Labels_All/MLS_all_platforms_AI_testing.shp";
    SAY_clipped_label_data = "geo_raw_data/SAY_clipped/Labels/SAY_Platform_polygons.shp";
    HNT_label_data = "geo_raw_data/HNT/Labels/HNT_Platform_OTLs_2023.shp";

    KOM_matlab_gt_labels_all_filename = "KOM/KOM_gt_platform.mat";
    MLS_matlab_gt_labels_all_filename = "MLS/MLS_gt_platform.mat";
    SAY_clipped_matlab_gt_labels_all_filename = "SAY_clipped/SAY_clipped_gt_platform.mat";
    HNT_matlab_gt_labels_all_filename = "HNT/HNT_4-5km2_gt_platform.mat";
end


%%%%%%% ALS 3-band data (11-02-2023) %%%%%%%
KOM_gis_geo_tiff_data = ["geo_raw_data/KOM/Rasters/kom_dsm_lidar_SVF_R20_D16_NRhigh.tif", "geo_raw_data/KOM/Rasters/kom_dsm_lidar_OPEN-POS_R20_D16_NRhigh.tif", ...
    "geo_raw_data/KOM/Rasters/kom_dsm_lidar_SLOPE.tif"];
MLS_gis_geo_tiff_data = ["geo_raw_data/MLS/Rasters/MLS_DEM_SVF_R20_D16_NRhigh.tif", "geo_raw_data/MLS/Rasters/MLS_DEM_OPEN-POS_R20_D16_NRhigh.tif", ...
    "geo_raw_data/MLS/Rasters/MLS_DEM_SLOPE.tif"];
SAY_clipped_gis_geo_tiff_data = ["geo_raw_data/SAY_clipped/Rasters/SAY_site_DEM_clipped_SVF_R20_D16_NRhigh.tif", ...
    "geo_raw_data/SAY_clipped/Rasters/SAY_site_DEM_clipped_OPEN-POS_R20_D16_NRhigh.tif", ...
    "geo_raw_data/SAY_clipped/Rasters/SAY_site_DEM_clipped_SLOPE.tif"];
HNT_gis_geo_tiff_data = ["geo_raw_data/HNT/Rasters/HNT_4-5km2_clip_SVF_R20_D16_NRhigh.tif", "geo_raw_data/HNT/Rasters/HNT_4-5km2_clip_OPEN-POS_R20_D16_NRhigh.tif", ...
    "geo_raw_data/HNT/Rasters/HNT_4-5km2_clip_SLOPE.tif"];
gis_output_filename = ["KOM/KOM_ALS_data.png", "MLS/MLS_ALS_data.png", "SAY_clipped/SAY_clipped_ALS_data.png", "HNT/HNT_4-5km2_ALS_data.png"];
matlab_data_filename = ["KOM/KOM_ALS_data.mat", "MLS/MLS_ALS_data.mat", "SAY_clipped/SAY_clipped_ALS_data.mat", "HNT/HNT_4-5km2_ALS_data.mat"];

%%%%%%% ALS 3-band data (11-02-2023) %%%%%%%

%%%%%%% DEM data %%%%%%%
% KOM_gis_geo_tiff_data = "geo_raw_data/KOM/Rasters/kom_dsm_lidar_Custom_flat_8bit.tif";
% MLS_gis_geo_tiff_data = "geo_raw_data/MLS/Rasters/MLS_DEM_Custom_flat_8bit.tif";
% SAY_clipped_gis_geo_tiff_data = "geo_raw_data/SAY_clipped/Rasters/SAY_site_DEM_clipped_Custom_flat_8bit.tif";
% HNT_gis_geo_tiff_data = "geo_raw_data/HNT/Rasters/HNT_4-5km2_clip_Custom_flat_8bit.tif";
% gis_output_filename = ["KOM/KOM_DEM_data.png", "MLS/MLS_DEM_data.png", "SAY_clipped/SAY_clipped_DEM_data.png", "HNT/HNT_4-5km2_DEM_data.png"];
% matlab_data_filename = ["KOM/KOM_DEM_data.mat", "MLS/MLS_DEM_data.mat", "SAY_clipped/SAY_clipped_DEM_data.mat", "HNT/HNT_4-5km2_DEM_data.mat"];
%%%%%%% DEM data %%%%%%%

%%%%%%% Hilshade data %%%%%%%
% KOM_gis_geo_tiff_data = "geo_raw_data/KOM/Rasters/KOM_HS.tif";
% MLS_gis_geo_tiff_data = "geo_raw_data/MLS/Rasters/MLS_HS.tif";
% SAY_clipped_gis_geo_tiff_data = "geo_raw_data/SAY_clipped/Rasters/SAY_site_HS.tif";
% HNT_gis_geo_tiff_data = "geo_raw_data/HNT/Rasters/HNT_4-5km2_clip_HS.tif";
% gis_output_filename = ["KOM/KOM_HS_data.png", "MLS/MLS_HS_data.png", "SAY_clipped/SAY_clipped_HS_data.png", "HNT/HNT_4-5km2_HS_data.png"];
% matlab_data_filename = ["KOM/KOM_HS_data.mat", "MLS/MLS_HS_data.mat", "SAY_clipped/SAY_clipped_HS_data.mat", "HNT/HNT_4-5km2_HS_data.mat"];
%%%%%%% Hilshade data %%%%%%%

switch DATASETINDEX
    case 1
        importData(1).filename = KOM_label_data(1);
        importData(1).labelValue = 1;
        
    case 2 
        importData(1).filename = MLS_label_data(1);
        importData(1).labelValue = 1;
    
    case 3 
        importData(1).filename = SAY_clipped_label_data(1);
        importData(1).labelValue = 1;
        
    case 4 
        importData(1).filename = HNT_label_data(1);
        importData(1).labelValue = 1;

    otherwise
        printf(1,"Error\n");
        return;
end

geotiff_data = [];
image_geo_output = [];
for gis_geo_tiff_data_index=1:length(KOM_gis_geo_tiff_data) 
    switch DATASETINDEX
        case 1
            gis_geotiff_filename = KOM_gis_geo_tiff_data(gis_geo_tiff_data_index);
            matlab_gt_labels_all_filename = KOM_matlab_gt_labels_all_filename; 
        case 2
            gis_geotiff_filename = MLS_gis_geo_tiff_data(gis_geo_tiff_data_index);
            matlab_gt_labels_all_filename = MLS_matlab_gt_labels_all_filename;
        case 3
            gis_geotiff_filename = SAY_clipped_gis_geo_tiff_data(gis_geo_tiff_data_index);
            matlab_gt_labels_all_filename = SAY_clipped_matlab_gt_labels_all_filename;
        case 4
            gis_geotiff_filename = HNT_gis_geo_tiff_data(gis_geo_tiff_data_index);
            matlab_gt_labels_all_filename = HNT_matlab_gt_labels_all_filename;
        otherwise
            fprintf(1,"Error\n");
            return;
    end
    gis_geotiff_info = geotiffinfo(gis_geotiff_filename);
    gis_geotiff_data = readgeoraster(gis_geotiff_filename);
    image_size = size(gis_geotiff_data);
    
    % manually handle bad elevation data (this is for old non-ALS GEO data)
    if (strcmp(gis_geotiff_filename,'../MLS/MLS_DEM.tif') == 1 || ...
        strcmp(gis_geotiff_filename,'../KOM/kom_dsm_lidar.tif') == 1)
        bad_pixel_values = max(gis_geotiff_data(:));
        artificial_min_value = min(gis_geotiff_data(:))-0.1;
        gis_geotiff_data(gis_geotiff_data==bad_pixel_values)=artificial_min_value;
    end

    gis_geotiff_data = double(gis_geotiff_data);
    % normalize data to the 0-MAX_INTENSITY intensity range
    geotiff_data = cat(3, geotiff_data, gis_geotiff_data);
    minValue = min(gis_geotiff_data(:));
    maxValue = max(gis_geotiff_data(:));
    range = maxValue - minValue;
    geo_tiff_image = uint8(MAX_INTENSITY*(gis_geotiff_data-minValue)/range);
    figure(gis_geo_tiff_data_index), imshow(geo_tiff_image,[]);
    image_geo_output = cat(3, image_geo_output, geo_tiff_image);
end
save(matlab_data_filename(DATASETINDEX), 'geotiff_data','-v7','-nocompression');
figure(4), imshow(image_geo_output,[]);
imwrite(image_geo_output, gis_output_filename(DATASETINDEX));

if isfile(matlab_gt_labels_all_filename)
    load(matlab_gt_labels_all_filename);
end

% write a for loop to iterate two classes. If only one classes is
% considered, set the iteration range to accomendate.
for shapefileIndex=1:length(importData)     % 1 is annular structure; 2 is platform.    
    gis_esri_shapefilename = importData(shapefileIndex).filename;
    if(strcmp(gis_esri_shapefilename, "NONE") == 1)
        labelInfo = struct('ID', {}, 'label_value', {}, 'label_name', {}, 'vertices', {}, 'center', {});
        all_labels(shapefileIndex).labels = labelInfo;
        continue;
    else
        shapefile_structure = shapeinfo(gis_esri_shapefilename);
        shapefile_data = shaperead(gis_esri_shapefilename);
        num_regions = length(shapefile_data);
        shp_range_x = shapefile_structure(1).BoundingBox(2,1) - shapefile_structure(1).BoundingBox(1,1);
        shp_range_y = shapefile_structure(1).BoundingBox(2,2) - shapefile_structure(1).BoundingBox(1,2);
    end

    WINDOWSIZE = region(shapefileIndex).WINDOWSIZE;

    image_size=size(geotiff_data);
    x0 = gis_geotiff_info.SpatialRef.XWorldLimits(1);
    y0 = gis_geotiff_info.SpatialRef.YWorldLimits(2);
    range_x = gis_geotiff_info.SpatialRef.XWorldLimits(2) - gis_geotiff_info.SpatialRef.XWorldLimits(1);
    range_y = gis_geotiff_info.SpatialRef.YWorldLimits(2) - gis_geotiff_info.SpatialRef.YWorldLimits(1);


    %labelInfo = struct('ID', {}, 'label_value', {}, 'label_name', {}, 'vertices', {}, 'center', {});
    %newRegionIdx = 1;   
    % start to write data from the beginning if labelInfo is emtpy
    if (exist('all_labels','var') == 0)
        labelInfo = struct('ID', {}, 'label_value', {}, 'label_name', {}, 'vertices', {}, 'center', {});
        newRegionIdx = 1;   % start to write data from the beginning if labelInfo is emtpy
    else
        if (length(all_labels) >= shapefileIndex)
            labelInfo = all_labels(shapefileIndex).labels;
            newRegionIdx = length(labelInfo) + 1;  % append the data to the end
        else
            labelInfo = struct('ID', {}, 'label_value', {}, 'label_name', {}, 'vertices', {}, 'center', {});
            newRegionIdx = 1;   % start to write data from the beginning if labelInfo is emtpy
        end            
    end

    empty_name_cnt = 0;
    for regionIdx=1:num_regions
        coords_x = image_size(2)*(shapefile_data(regionIdx).X - x0)./range_x;
        coords_x(isnan(coords_x))=coords_x(1);
        coords_y = image_size(1)*(y0 - shapefile_data(regionIdx).Y)./range_y;
        coords_y(isnan(coords_y))=coords_y(1);
        polygon_vertices = [coords_x', coords_y'];
        if(isempty(coords_x))
            sprintf('Skipping an empty feature.\n');
            continue
        end
        %polygon_vertices(any(isnan(polygon_vertices), 2), :) = [coords_x(1), coords_y(1)];
        xy_region_min = min(polygon_vertices,[],1);
        xy_region_max = max(polygon_vertices,[],1);
        xy_region_range = xy_region_max - xy_region_min;
        xy_region_center = mean(polygon_vertices,1);

        %xy_region_center_pixel_id = round(xy_region_center(1)/20)*image_size(1) + round(xy_region_center(2)/20);
        xy_region_center_pixel_id = shapefile_data(regionIdx).str_name;
        
        % Assign IDs to regions without a ID name. Otherwise the ID-based
        % matching and replacing later will cause a problem
        if(isempty(xy_region_center_pixel_id))
            xy_region_center_pixel_id = strcat("empty_name", num2str(empty_name_cnt));
            empty_name_cnt = empty_name_cnt + 1;
        end

        % if region size is 0 --> this is a point feature  
        if (prod(xy_region_range) == 0)
            fprintf(1,'Cannot import shapefile with point features.\n');
            return;
        end

        % search for an existing record for this region
        matchFound = false; 
        for matchedRegionIdx=1:length(labelInfo)
            if (strcmp(labelInfo(matchedRegionIdx).ID, xy_region_center_pixel_id))
                str_out = sprintf('Replaced existing region data for ID = %s.\n', labelInfo(matchedRegionIdx).ID);
                fprintf(1, str_out);
                labelInfo(matchedRegionIdx).ID = xy_region_center_pixel_id;
                labelInfo(matchedRegionIdx).label_value = region(shapefileIndex).LabelValue;
                labelInfo(matchedRegionIdx).label_name = region(shapefileIndex).Name;
                labelInfo(matchedRegionIdx).vertices = polygon_vertices;
                labelInfo(matchedRegionIdx).center = xy_region_center;
                matchFound = true;
                break;
            end
        end
        
        if (~matchFound)
            str_out = sprintf('Added new region data for ID = %s.\n', xy_region_center_pixel_id);
            fprintf(1, str_out);
            labelInfo(newRegionIdx).ID = xy_region_center_pixel_id;
            labelInfo(newRegionIdx).label_value = region(shapefileIndex).LabelValue;
            labelInfo(newRegionIdx).label_name = region(shapefileIndex).Name;
            labelInfo(newRegionIdx).vertices = polygon_vertices;
            labelInfo(newRegionIdx).center = xy_region_center;              
            newRegionIdx = newRegionIdx + 1;
        end


        %   'MarkerSize', 5, ...
    %     figure(1), hold on, drawpolygon('Position', polygon_vertices, ...
    %         'LineWidth',1,'FaceAlpha', 0.3, 'Color', region(shapefileIndex).Color, ...
    %         'SelectedColor', region(shapefileIndex).Color);
        %   'MarkerSize', 5, ...
        figure(4), hold on, drawpolygon('Position', polygon_vertices, ...
            'LineWidth',0.5,'FaceAlpha', 0.3, 'Color', region(shapefileIndex).Color, ...
            'SelectedColor', region(shapefileIndex).Color);

    %     if (PLOT_REGION_MESH)
    %         PLOT_MARGIN = 10;
    %         x_coord_list = (xy_region_min(1)-PLOT_MARGIN/2):(xy_region_max(1)+PLOT_MARGIN/2);
    %         x_coord_list(x_coord_list < 0) = [];
    %         x_coord_list(x_coord_list > image_size(2)) = [];
    %         y_coord_list = (xy_region_min(2)-PLOT_MARGIN/2):(xy_region_max(2)+PLOT_MARGIN/2);
    %         y_coord_list(y_coord_list < 0) = [];
    %         y_coord_list(y_coord_list > image_size(1)) = [];
    %         xx_vals = int32(x_coord_list);
    %         yy_vals = int32(y_coord_list);
    %         zz_vals = geotiff_data(yy_vals,xx_vals);
    %         [x_meshgrid, y_meshgrid] = meshgrid(xx_vals,yy_vals);
    %         titlestr = sprintf('%s region index %d', region(shapefileIndex).Name, regionIdx);
    %         figure(2), hold off, mesh(x_meshgrid, y_meshgrid, zz_vals), title(titlestr);
    %         %     figure(2), view(0,90)
    %         %     [new_centerpt_x, new_centerpt_y] = ginput(1);
    %         %     coords_x = coords_x - centerpt(1) + new_centerpt_x;
    %         %     coords_y = coords_y - centerpt(2) + new_centerpt_y;
    %         % save the new coordinates
    %         %pause(0.5);
    %     end    

        num_vertices = size(polygon_vertices,1);

        %labelInfo = all_labels(shapefileIndex).labels;

    end
    all_labels(shapefileIndex).labels = labelInfo;
end

save(matlab_gt_labels_all_filename, 'all_labels','-v7','-nocompression');