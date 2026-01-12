use std::num::NonZero;

use async_channel::{Receiver, Sender};
use bevy::asset::{RenderAssetUsages, embedded_asset, load_embedded_asset};
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::ScheduleSystem;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{self, RenderGraph, RenderLabel};
use bevy::render::render_resource::binding_types::{
    storage_buffer, storage_buffer_read_only, storage_buffer_sized, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntryBuilder, Buffer, BufferUsages, CachedComputePipelineId,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType,
    StorageBuffer, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::{Render, RenderApp, RenderSystems};

pub struct MarchingCubesPlugin<Sampler, Material> {
    _marker: std::marker::PhantomData<(Sampler, Material)>,
}

impl<Sampler, Material> Default for MarchingCubesPlugin<Sampler, Material> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Sampler: ChunkComputeShader + Send + Sync + 'static, Material: Asset + bevy::prelude::Material>
    Plugin for MarchingCubesPlugin<Sampler, Material>
{
    fn build(&self, app: &mut App) {
        {
            embedded_asset!(app, "marching_cubes.wgsl");
        }

        app.add_plugins((
            ExtractResourcePlugin::<ChunkGeneratorSettings<Sampler>>::default(),
            ExtractComponentPlugin::<ChunkRenderData<Sampler>>::default(),
        ))
        .init_resource::<ChunkGeneratorCache<Sampler>>()
        .init_resource::<ChunkGeneratorNodeNextId>()
        .add_systems(
            Update,
            (
                check_run_done::<Sampler>,
                update_chunk_loaders::<Sampler>,
                queue_chunks::<Sampler>,
                start_chunks::<Sampler, Material>,
            )
                .chain()
                .in_set(ChunkGenSystems)
                .run_if(resource_exists::<ChunkGeneratorCache<Sampler>>),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            (
                init_compute_pipelines::<Sampler>
                    .in_set(RenderSystems::PrepareResources)
                    .run_if(not(resource_exists::<
                        ChunkGeneratorComputePipelines<Sampler>,
                    >)),
                (
                    Sampler::prepare_extra_buffers(),
                    prepare_bind_groups::<Sampler>,
                )
                    .in_set(RenderSystems::PrepareBindGroups),
                remove_nodes::<Sampler>.in_set(RenderSystems::Cleanup),
            ),
        );
    }
}

#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ChunkGenSystems;

const WORKGROUP_SIZE: u32 = 8;

#[derive(ShaderType)]
struct MeshSettings {
    num_voxels_per_axis: u32,
    num_samples_per_axis: u32,
    chunk_size: f32,
    surface_threshold: f32,
}

#[derive(ShaderType, Default, Debug, Clone, Copy, Reflect)]
struct Vertex {
    position: Vec3,
    _padding1: f32,
    normal: Vec3,
    _padding2: f32,
}

#[derive(ShaderType, Default, Debug, Clone, Copy, Reflect)]
struct Triangle {
    vertex_a: u32,
    vertex_b: u32,
    vertex_c: u32,
}

#[derive(Resource, Default)]
struct ChunkGeneratorNodeNextId(usize);

impl ChunkGeneratorNodeNextId {
    fn next(&mut self) -> usize {
        let next = self.0;
        self.0 += 1;
        next
    }
}

#[derive(RenderLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct ChunkGeneratorNodeLabel(usize);

struct ChunkGeneratorNode<Sampler> {
    tx: Sender<ChunkGeneratorRun>,
    sample_workgroups: u32,
    sample_bind_group: BindGroup,
    march_workgroups: u32,
    march_bind_group: BindGroup,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler: Send + Sync + 'static> render_graph::Node for ChunkGeneratorNode<Sampler> {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        info!("Rendering a chunk");

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.resource::<ChunkGeneratorComputePipelines<Sampler>>();

        {
            let Some(sample_pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.sample_pipeline)
            else {
                return Ok(());
            };

            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("marching cubes sample pass"),
                        ..default()
                    });

            pass.set_bind_group(0, &self.sample_bind_group, &[]);
            pass.set_pipeline(sample_pipeline);
            pass.dispatch_workgroups(
                self.sample_workgroups,
                self.sample_workgroups,
                self.sample_workgroups,
            );
        }

        {
            let Some(march_pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.march_pipeline)
            else {
                return Ok(());
            };

            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("marching cubes march pass"),
                        ..default()
                    });

            pass.set_bind_group(0, &self.march_bind_group, &[]);
            pass.set_pipeline(march_pipeline);
            pass.dispatch_workgroups(
                self.march_workgroups,
                self.march_workgroups,
                self.march_workgroups,
            );
        }

        info!("Done rendering a chunk");

        let _ = self.tx.force_send(ChunkGeneratorRun);

        Ok(())
    }
}

struct ChunkGeneratorRun;

#[derive(Resource)]
struct ChunkGeneratorComputePipelines<Sampler> {
    sample_layout: BindGroupLayout,
    sample_pipeline: CachedComputePipelineId,
    march_layout: BindGroupLayout,
    march_pipeline: CachedComputePipelineId,
    _marker: std::marker::PhantomData<Sampler>,
}

fn init_compute_pipelines<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    settings: Res<ChunkGeneratorSettings<Sampler>>,
) {
    info!("Init compute pipelines");

    let sample_layout = render_device.create_bind_group_layout(
        "marching cubes sample bind group",
        &[
            uniform_buffer::<IVec3>(false),
            uniform_buffer::<MeshSettings>(false),
            storage_buffer::<Vec<f32>>(false),
        ]
        .into_iter()
        .chain(Sampler::define_extra_buffers())
        .enumerate()
        .map(|(i, b)| b.build(i as u32, ShaderStages::COMPUTE))
        .collect::<Vec<_>>(),
    );
    let sample_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("marching cubes sample compute shader".into()),
        layout: vec![sample_layout.clone()],
        shader: asset_server.load(Sampler::shader_path()),
        ..default()
    });

    let march_layout = render_device.create_bind_group_layout(
        "marching cubes march bind group",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<Vec<f32>>(false),
                uniform_buffer::<MeshSettings>(false),
                storage_buffer_sized(false, Some(settings.vertices_buffer_size())),
                storage_buffer::<u32>(false),
                storage_buffer_sized(false, Some(settings.triangles_buffer_size())),
                storage_buffer::<u32>(false),
            ),
        ),
    );
    let march_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("marching cubes march compute shader".into()),
        layout: vec![march_layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "marching_cubes.wgsl"),
        entry_point: Some("main".into()),
        ..default()
    });

    commands.insert_resource(ChunkGeneratorComputePipelines::<Sampler> {
        sample_layout,
        sample_pipeline,
        march_layout,
        march_pipeline,
        _marker: default(),
    });
}

pub trait ChunkComputeShader {
    // TODO: this is just left over from bevy_app_compute. what's the bevy way to pass in an asset? probably a resource, same as settings
    fn shader_path() -> String;
    fn prepare_extra_buffers() -> ScheduleConfigs<ScheduleSystem> {
        IntoSystem::into_system(|| {}).into_configs()
    }
    fn define_extra_buffers() -> Vec<BindGroupLayoutEntryBuilder> {
        vec![]
    }
}

#[derive(Component, Debug)]
pub struct Chunk<Sampler> {
    pub position: IVec3,
    _marker: std::marker::PhantomData<Sampler>,
}

#[derive(Component, Debug)]
struct ChunkGenData {
    rx: Receiver<ChunkGeneratorRun>,
    vertices: Option<Vec<Vertex>>,
    triangles: Option<Vec<Triangle>>,
}

#[derive(Component, ExtractComponent, Debug)]
pub struct ChunkRenderData<Sampler: Send + Sync + 'static> {
    position: IVec3,
    node_id: usize,
    tx: Sender<ChunkGeneratorRun>,
    vertices_buffer: Handle<ShaderStorageBuffer>,
    num_vertices_buffer: Handle<ShaderStorageBuffer>,
    triangles_buffer: Handle<ShaderStorageBuffer>,
    num_triangles_buffer: Handle<ShaderStorageBuffer>,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler: Send + Sync + 'static> Clone for ChunkRenderData<Sampler> {
    fn clone(&self) -> Self {
        Self {
            position: self.position,
            node_id: self.node_id,
            tx: self.tx.clone(),
            vertices_buffer: self.vertices_buffer.clone(),
            num_vertices_buffer: self.num_vertices_buffer.clone(),
            triangles_buffer: self.triangles_buffer.clone(),
            num_triangles_buffer: self.num_triangles_buffer.clone(),
            _marker: self._marker,
        }
    }
}

#[derive(Component, Default, Debug)]
pub struct ChunkLoader<T> {
    pub position: IVec3,
    pub loading_radius: i32,
    _marker: std::marker::PhantomData<T>,
}

impl<T> ChunkLoader<T> {
    pub fn new(loading_radius: i32) -> Self {
        Self {
            position: IVec3::ZERO,
            loading_radius,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Resource, Debug)]
pub struct ChunkMaterial<Sampler, Material: Asset> {
    pub material: Handle<Material>,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler, Material: Asset> ChunkMaterial<Sampler, Material> {
    pub fn new(material: Handle<Material>) -> Self {
        Self {
            material,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Resource, ExtractResource, Debug)]
pub struct ChunkGeneratorSettings<T: Send + Sync + 'static> {
    surface_threshold: f32,
    num_voxels_per_axis: u32,
    chunk_size: f32,
    bounds: Option<GenBounds>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Send + Sync + 'static> Clone for ChunkGeneratorSettings<T> {
    fn clone(&self) -> Self {
        Self {
            surface_threshold: self.surface_threshold,
            num_voxels_per_axis: self.num_voxels_per_axis,
            chunk_size: self.chunk_size,
            bounds: self.bounds.clone(),
            _marker: self._marker,
        }
    }
}

#[derive(Debug, Clone)]
struct GenBounds {
    min: Vec3,
    max: Vec3,
}

impl<T: Send + Sync + 'static> ChunkGeneratorSettings<T> {
    pub fn new(num_voxels_per_axis: u32, chunk_size: f32) -> Self {
        Self {
            surface_threshold: 0.0,
            num_voxels_per_axis,
            chunk_size,
            bounds: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_surface_threshold(mut self, surface_threshold: f32) -> Self {
        self.surface_threshold = surface_threshold;
        self
    }

    pub fn with_bounds(mut self, min: Vec3, max: Vec3) -> Self {
        self.bounds = Some(GenBounds { min, max });
        self
    }

    pub fn num_samples_per_axis(&self) -> u32 {
        self.num_voxels_per_axis + 3 // We sample the next chunk over too for normals
    }

    pub fn max_num_vertices(&self) -> u64 {
        self.max_num_triangles() * 3
    }

    pub fn vertices_buffer_size(&self) -> NonZero<u64> {
        (size_of::<Vertex>() as u64 * self.max_num_vertices())
            .try_into()
            .expect("zero vertices")
    }

    pub fn max_num_triangles(&self) -> u64 {
        (self.num_voxels_per_axis as u64).pow(3) * 5
    }

    pub fn triangles_buffer_size(&self) -> NonZero<u64> {
        (size_of::<Triangle>() as u64 * self.max_num_triangles())
            .try_into()
            .expect("zero triangles")
    }

    pub fn voxel_size(&self) -> f32 {
        self.chunk_size / self.num_voxels_per_axis as f32
    }

    pub fn position_to_chunk(&self, position: Vec3) -> IVec3 {
        (position / self.chunk_size).floor().as_ivec3()
    }

    pub fn chunk_to_position(&self, chunk: IVec3) -> Vec3 {
        chunk.as_vec3() * self.chunk_size
    }

    fn is_chunk_in_bounds(&self, chunk_position: IVec3) -> bool {
        if let Some(bounds) = &self.bounds {
            let position = self.chunk_to_position(chunk_position);
            position.x >= bounds.min.x
                && position.x <= bounds.max.x
                && position.y >= bounds.min.y
                && position.y <= bounds.max.y
                && position.z >= bounds.min.z
                && position.z <= bounds.max.z
        } else {
            true
        }
    }
}

#[derive(Resource, Debug, Clone)]
pub struct ChunkGeneratorCache<T> {
    loaded_chunks: HashMap<IVec3, LoadState>,
    chunks_to_load: Vec<IVec3>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Send + Sync + 'static> ChunkGeneratorCache<T> {
    pub fn is_chunk_marked(
        &self,
        settings: &ChunkGeneratorSettings<T>,
        chunk_position: IVec3,
    ) -> bool {
        !settings.is_chunk_in_bounds(chunk_position)
            || self.loaded_chunks.contains_key(&chunk_position)
    }

    pub fn is_chunk_generated(
        &self,
        settings: &ChunkGeneratorSettings<T>,
        chunk_position: IVec3,
    ) -> bool {
        !settings.is_chunk_in_bounds(chunk_position)
            || matches!(
                self.loaded_chunks.get(&chunk_position),
                Some(LoadState::Finished)
            )
    }

    pub fn is_chunk_with_position_marked(
        &self,
        settings: &ChunkGeneratorSettings<T>,
        position: Vec3,
    ) -> bool {
        self.is_chunk_marked(settings, settings.position_to_chunk(position))
    }

    pub fn is_chunk_with_position_generated(
        &self,
        settings: &ChunkGeneratorSettings<T>,
        position: Vec3,
    ) -> bool {
        self.is_chunk_generated(settings, settings.position_to_chunk(position))
    }
}

impl<T> Default for ChunkGeneratorCache<T> {
    fn default() -> Self {
        Self {
            loaded_chunks: default(),
            chunks_to_load: default(),
            _marker: default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LoadState {
    Loading,
    Finished,
}

fn update_chunk_loaders<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut chunk_loaders: Query<
        (&mut ChunkLoader<Sampler>, &GlobalTransform),
        Changed<GlobalTransform>,
    >,
) {
    for (mut chunk_loader, transform) in chunk_loaders.iter_mut() {
        let chunk_position = (transform.translation() / settings.chunk_size)
            .floor()
            .as_ivec3();

        // Properly update change detection
        if chunk_loader.position != chunk_position {
            chunk_loader.position = chunk_position;
        }
    }
}

fn queue_chunks<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut cache: ResMut<ChunkGeneratorCache<Sampler>>,
    chunk_loaders: Query<&ChunkLoader<Sampler>, Changed<ChunkLoader<Sampler>>>,
) {
    for chunk_loader in chunk_loaders.iter() {
        let mut load_order = Vec::new();
        for x in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
            for y in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                for z in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                    load_order.push(Vec3::new(x as f32, y as f32, z as f32));
                }
            }
        }

        load_order.sort_by(|a, b| {
            // Sort ascending so that the closest chunks are loaded first
            a.length_squared()
                .partial_cmp(&b.length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for offset in load_order {
            let chunk_position = chunk_loader.position + offset.as_ivec3();
            if !cache.is_chunk_marked(&settings, chunk_position) {
                cache
                    .loaded_chunks
                    .insert(chunk_position, LoadState::Loading);
                cache.chunks_to_load.push(chunk_position);
                info!("Queued chunk for loading: {chunk_position:?}");
            }
        }
    }
}

fn start_chunks<
    Sampler: ChunkComputeShader + Send + Sync + 'static,
    Material: Asset + bevy::prelude::Material,
>(
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut ids: ResMut<ChunkGeneratorNodeNextId>,
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut cache: ResMut<ChunkGeneratorCache<Sampler>>,
    material: Res<ChunkMaterial<Sampler, Material>>,
) {
    for chunk_position in cache.chunks_to_load.drain(..) {
        let max_num_vertices = settings.max_num_vertices();
        let max_num_triangles = settings.max_num_triangles();
        let vertices_buffer_size: u64 = settings.vertices_buffer_size().into();
        let triangles_buffer_size: u64 = settings.triangles_buffer_size().into();

        info!("start_chunks {chunk_position:?} {max_num_vertices} {max_num_triangles}");

        let mut vertices_buffer = ShaderStorageBuffer::with_size(
            vertices_buffer_size as usize,
            RenderAssetUsages::default(),
        );
        vertices_buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
        let vertices_buffer = buffers.add(vertices_buffer);
        let mut num_vertices_buffer = ShaderStorageBuffer::from(0u32);
        num_vertices_buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
        let num_vertices_buffer = buffers.add(num_vertices_buffer);
        let mut triangles_buffer = ShaderStorageBuffer::with_size(
            triangles_buffer_size as usize,
            RenderAssetUsages::default(),
        );
        triangles_buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
        let triangles_buffer = buffers.add(triangles_buffer);
        let mut num_triangles_buffer = ShaderStorageBuffer::from(0u32);
        num_triangles_buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
        let num_triangles_buffer = buffers.add(num_triangles_buffer);

        let node_id = ids.next();
        let (tx, rx) = async_channel::bounded(1);

        let chunk_entity = commands
            .spawn((
                Name::new(format!("Chunk {chunk_position:?}")),
                Transform::from_translation(settings.chunk_to_position(chunk_position)),
                MeshMaterial3d(material.material.clone()),
                Chunk::<Sampler> {
                    position: chunk_position,
                    _marker: default(),
                },
                ChunkGenData {
                    rx,
                    vertices: None,
                    triangles: None,
                },
                ChunkRenderData::<Sampler> {
                    position: chunk_position,
                    node_id,
                    tx,
                    vertices_buffer: vertices_buffer.clone(),
                    num_vertices_buffer: num_vertices_buffer.clone(),
                    triangles_buffer: triangles_buffer.clone(),
                    num_triangles_buffer: num_triangles_buffer.clone(),
                    _marker: default(),
                },
            ))
            .observe(finish_chunk::<Sampler>)
            .id();

        commands
            .spawn((
                Name::new(format!("Chunk {chunk_position:?} num_vertices readback")),
                Readback::buffer(num_vertices_buffer),
                ChildOf(chunk_entity),
            ))
            .observe(
                move |readback: On<ReadbackComplete>,
                      mut chunks: Query<(&mut ChunkGenData, Has<ChunkRenderData<Sampler>>)>,
                      mut commands: Commands|
                      -> Result {
                    let (mut chunk, not_done) = chunks.get_mut(chunk_entity)?;
                    if not_done {
                        return Ok(());
                    }
                    let num_vertices: u32 = readback.to_shader_type();
                    info!("num_vertices readback {chunk_position:?} {num_vertices}");
                    commands.entity(readback.entity).despawn();
                    if num_vertices > 0 {
                        commands
                            .spawn((
                                Name::new(format!("Chunk {chunk_position:?} vertices readback")),
                                Readback::buffer_range(
                                    vertices_buffer.clone(),
                                    0,
                                    size_of::<Vertex>() as u64 * num_vertices as u64,
                                ),
                                ChildOf(chunk_entity),
                            ))
                            .observe(
                                move |readback: On<ReadbackComplete>,
                                      mut chunks: Query<&mut ChunkGenData>,
                                      mut commands: Commands|
                                      -> Result {
                                    let vertices: Vec<Vertex> = readback.to_shader_type();
                                    info!(
                                        "vertices readback {chunk_position:?} {}",
                                        vertices.len()
                                    );
                                    let mut chunk = chunks.get_mut(chunk_entity)?;
                                    chunk.vertices = Some(vertices);
                                    commands.trigger(ReadbackReallyComplete(chunk_entity));
                                    commands.entity(readback.entity).despawn();
                                    Ok(())
                                },
                            );
                    } else {
                        chunk.vertices = Some(vec![]);
                        commands.trigger(ReadbackReallyComplete(chunk_entity));
                    }
                    Ok(())
                },
            );

        commands
            .spawn((
                Name::new(format!("Chunk {chunk_position:?} num_triangles readback")),
                Readback::buffer(num_triangles_buffer),
                ChildOf(chunk_entity),
            ))
            .observe(
                move |readback: On<ReadbackComplete>,
                      mut chunks: Query<(&mut ChunkGenData, Has<ChunkRenderData<Sampler>>)>,
                      mut commands: Commands|
                      -> Result {
                    let (mut chunk, not_done) = chunks.get_mut(chunk_entity)?;
                    if not_done {
                        return Ok(());
                    }
                    let num_triangles: u32 = readback.to_shader_type();
                    info!("num_triangles readback {chunk_position:?} {num_triangles}");
                    commands.entity(readback.entity).despawn();
                    if num_triangles > 0 {
                        commands
                            .spawn((
                                Name::new(format!("Chunk {chunk_position:?} triangles readback")),
                                Readback::buffer_range(
                                    triangles_buffer.clone(),
                                    0,
                                    size_of::<Triangle>() as u64 * num_triangles as u64,
                                ),
                                ChildOf(chunk_entity),
                            ))
                            .observe(
                                move |readback: On<ReadbackComplete>,
                                      mut chunks: Query<&mut ChunkGenData>,
                                      mut commands: Commands|
                                      -> Result {
                                    let triangles: Vec<Triangle> = readback.to_shader_type();
                                    info!(
                                        "triangles readback {chunk_position:?} {}",
                                        triangles.len()
                                    );
                                    let mut chunk = chunks.get_mut(chunk_entity)?;
                                    chunk.triangles = Some(triangles);
                                    commands.trigger(ReadbackReallyComplete(chunk_entity));
                                    commands.entity(readback.entity).despawn();
                                    Ok(())
                                },
                            );
                    } else {
                        chunk.triangles = Some(vec![]);
                        commands.trigger(ReadbackReallyComplete(chunk_entity));
                    }
                    Ok(())
                },
            );
    }
}

fn check_run_done<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    chunks: Query<(Entity, &ChunkGenData), With<ChunkRenderData<Sampler>>>,
    mut commands: Commands,
) {
    for (entity, chunk) in chunks.iter() {
        if chunk.rx.try_recv().is_ok() {
            commands.entity(entity).remove::<ChunkRenderData<Sampler>>();
        }
    }
}

#[derive(Component)]
pub struct ChunkRenderExtraBuffers {
    pub buffers: Vec<Buffer>,
}

fn prepare_bind_groups<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    render_device: Res<RenderDevice>,
    mut render_graph: ResMut<RenderGraph>,
    render_queue: Res<RenderQueue>,
    chunks: Query<(&ChunkRenderData<Sampler>, Option<&ChunkRenderExtraBuffers>)>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    pipelines: Res<ChunkGeneratorComputePipelines<Sampler>>,
) {
    let num_voxels_per_axis = settings.num_voxels_per_axis;
    let num_samples_per_axis = settings.num_samples_per_axis();
    let chunk_size = settings.chunk_size;
    let surface_threshold = settings.surface_threshold;
    let sample_workgroups = (num_samples_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;
    let march_workgroups = (num_voxels_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;

    for (chunk, extra_buffers) in chunks.iter() {
        info!("prepare_bind_groups {}", chunk.position);

        let mut chunk_position_buffer = UniformBuffer::from(chunk.position);
        chunk_position_buffer.write_buffer(&render_device, &render_queue);

        let mut settings_buffer = UniformBuffer::from(MeshSettings {
            num_voxels_per_axis,
            num_samples_per_axis,
            chunk_size,
            surface_threshold,
        });
        settings_buffer.write_buffer(&render_device, &render_queue);

        let mut densities_buffer = StorageBuffer::from(vec![
            0.0f32;
            settings.num_samples_per_axis().pow(3)
                as usize
        ]);
        densities_buffer.write_buffer(&render_device, &render_queue);

        let vertices_buffer = buffers.get(&chunk.vertices_buffer).unwrap();
        let num_vertices_buffer = buffers.get(&chunk.num_vertices_buffer).unwrap();
        let triangles_buffer = buffers.get(&chunk.triangles_buffer).unwrap();
        let num_triangles_buffer = buffers.get(&chunk.num_triangles_buffer).unwrap();

        let sample_bind_group = render_device.create_bind_group(
            Some("marching cubes sample bind group"),
            &pipelines.sample_layout,
            &[
                chunk_position_buffer.binding().unwrap(),
                settings_buffer.binding().unwrap(),
                densities_buffer.binding().unwrap(),
            ]
            .into_iter()
            .chain(
                extra_buffers
                    .map(|b| &b.buffers)
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|b| b.as_entire_binding()),
            )
            .enumerate()
            .map(|(i, res)| BindGroupEntry {
                binding: i as u32,
                resource: res,
            })
            .collect::<Vec<_>>(),
        );

        let march_bind_group = render_device.create_bind_group(
            Some("marching cubes march bind group"),
            &pipelines.march_layout,
            &BindGroupEntries::sequential((
                densities_buffer.binding().unwrap(),
                settings_buffer.binding().unwrap(),
                vertices_buffer.buffer.as_entire_binding(),
                num_vertices_buffer.buffer.as_entire_binding(),
                triangles_buffer.buffer.as_entire_binding(),
                num_triangles_buffer.buffer.as_entire_binding(),
            )),
        );

        // Should this be here?
        render_graph.add_node(
            ChunkGeneratorNodeLabel(chunk.node_id),
            ChunkGeneratorNode::<Sampler> {
                tx: chunk.tx.clone(),
                sample_workgroups,
                sample_bind_group,
                march_workgroups,
                march_bind_group,
                _marker: default(),
            },
        );
    }
}

#[derive(EntityEvent)]
struct ReadbackReallyComplete(Entity);

fn finish_chunk<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    readback: On<ReadbackReallyComplete>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cache: ResMut<ChunkGeneratorCache<Sampler>>,
    chunks: Query<(&Chunk<Sampler>, &ChunkGenData)>,
) -> Result {
    let Ok((
        &Chunk {
            position: chunk_position,
            ..
        },
        chunk,
    )) = chunks.get(readback.0)
    else {
        return Ok(()); // It's already done
    };

    info!("finish_chunk {chunk_position}");

    let Some(ref vertices) = chunk.vertices else {
        return Ok(());
    };
    let Some(ref triangles) = chunk.triangles else {
        return Ok(());
    };

    if !vertices.is_empty() && !triangles.is_empty() {
        let mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(bevy::mesh::Indices::U32(
            triangles
                .iter()
                .flat_map(|t| [t.vertex_c, t.vertex_b, t.vertex_a])
                .collect(),
        ))
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vertices.iter().map(|v| v.normal).collect::<Vec<_>>(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vertices.iter().map(|v| v.position.xy()).collect::<Vec<_>>(),
        );

        commands.entity(readback.0).insert(Mesh3d(meshes.add(mesh)));
    }
    cache
        .loaded_chunks
        .insert(chunk_position, LoadState::Finished);

    commands.entity(readback.0).remove::<ChunkGenData>();

    Ok(())
}

fn remove_nodes<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    chunks: Query<&ChunkRenderData<Sampler>>,
    mut render_graph: ResMut<RenderGraph>,
) -> Result {
    for chunk in chunks.iter() {
        info!("removing node {}", chunk.position);
        render_graph.remove_node(ChunkGeneratorNodeLabel(chunk.node_id))?;
    }
    Ok(())
}
